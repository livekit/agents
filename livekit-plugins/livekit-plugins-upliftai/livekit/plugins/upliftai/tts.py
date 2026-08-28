"""
Uplift TTS Plugin for LiveKit, this will soon be available as a python lib
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import socketio  # type: ignore[import-untyped, import-not-found, unused-ignore]

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import codecs, is_given

from .log import logger

# Output format options
OutputFormat = Literal[
    "PCM_22050_16",
    "WAV_22050_16",
    "WAV_22050_32",
    "MP3_22050_32",
    "MP3_22050_64",
    "MP3_22050_128",
    "OGG_22050_16",
    "ULAW_8000_8",
]

# Default configuration
DEFAULT_BASE_URL = "wss://api.upliftai.org"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_NUM_CHANNELS = 1
DEFAULT_VOICE_ID = "v_meklc281"
DEFAULT_OUTPUT_FORMAT: OutputFormat = "MP3_22050_32"
WEBSOCKET_NAMESPACE = "/text-to-speech/multi-stream"

# Sentence-ending punctuation: English, Urdu (۔ U+06D4, ؟ U+061F) and full-width marks
SENTENCE_END_CHARS = ".!?…۔؟。！？"
# closing quotes/brackets that may trail the sentence-ending punctuation
_CLOSING_CHARS = "\"'”’»)]"

DEFAULT_MIN_CHUNK_LEN = 20
DEFAULT_MAX_CHUNK_LEN = 200
# speaking-rate bounds enforced by the server (speed 1.0 = normal rate)
MIN_SPEED = 0.5
MAX_SPEED = 2.0
# how many synthesis requests may be in flight ahead of the one currently being emitted
MAX_PIPELINED_REQUESTS = 3
AUDIO_CHUNK_TIMEOUT = 30.0


def get_content_type_from_output_format(output_format: OutputFormat) -> str:
    """Get MIME type based on output format"""
    if output_format == "PCM_22050_16":
        return "audio/pcm"
    elif output_format == "WAV_22050_16":
        return "audio/wav"
    elif output_format == "WAV_22050_32":
        return "audio/wav"
    elif output_format.startswith("MP3"):
        return "audio/mpeg"
    elif output_format.startswith("OGG"):
        return "audio/ogg"
    elif output_format == "ULAW_8000_8":
        return "audio/x-mulaw"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _ends_sentence(token: str) -> bool:
    stripped = token.rstrip(_CLOSING_CHARS)
    return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS


def _validate_speed(speed: float) -> None:
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")


@dataclass
class VoiceSettings:
    """Voice configuration settings"""

    voice_id: str = DEFAULT_VOICE_ID
    output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT
    speed: float | None = None


@dataclass
class _TTSOptions:
    """Internal TTS options"""

    base_url: str
    api_key: str
    voice_settings: VoiceSettings
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    sample_rate: int
    num_channels: int
    phrase_replacement_config_id: str | None
    min_chunk_len: int
    max_chunk_len: int


class TTS(tts.TTS):
    """Uplift TTS implementation for LiveKit"""

    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice_id: str = DEFAULT_VOICE_ID,
        output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        phrase_replacement_config_id: NotGivenOr[str] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN,
        max_chunk_len: int = DEFAULT_MAX_CHUNK_LEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Uplift TTS.

        Args:
            base_url: Base URL for TTS service. Defaults to wss://api.upliftai.org
            api_key: API key for authentication
            voice_id: Voice ID to use. Defaults to "v_meklc281"
            output_format: Audio output format. Options:
                - 'PCM_22050_16': PCM format, 22.05kHz, 16-bit
                - 'WAV_22050_16': WAV format, 22.05kHz, 16-bit
                - 'WAV_22050_32': WAV format, 22.05kHz, 32-bit
                - 'MP3_22050_32': MP3 format, 22.05kHz, 32kbps (default)
                - 'MP3_22050_64': MP3 format, 22.05kHz, 64kbps
                - 'MP3_22050_128': MP3 format, 22.05kHz, 128kbps
                - 'OGG_22050_16': OGG format, 22.05kHz, 16-bit
                - 'ULAW_8000_8': μ-law format, 8kHz, 8-bit. Intended for telephony
                  integrations; not currently decodable in the agents audio pipeline
                  (headerless μ-law is not supported by the SDK decoder)
            num_channels: Number of audio channels. Defaults to 1 (mono)
            phrase_replacement_config_id: Optional ID for phrase replacement configuration
            word_tokenizer: Tokenizer for processing text. Defaults to `livekit.agents.tokenize.basic.WordTokenizer`.
            min_chunk_len: Minimum buffered characters before a sentence boundary
                triggers a synthesis request. Defaults to 20
            max_chunk_len: Maximum buffered characters before a synthesis request is
                forced, even without a sentence boundary. Defaults to 200
            speed: Speaking rate, 0.5 (half speed) to 2.0 (double speed).
                Defaults to the server's normal rate (1.0)
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=num_channels,
        )

        # Get configuration from environment if not provided
        resolved_base_url: str = (
            base_url
            if is_given(base_url)
            else os.environ.get("UPLIFTAI_BASE_URL", DEFAULT_BASE_URL)
        )
        resolved_api_key: str | None = (
            api_key if is_given(api_key) else os.environ.get("UPLIFTAI_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "API key is required, either as argument or set UPLIFTAI_API_KEY environment variable"
            )

        if min_chunk_len < 1:
            raise ValueError("min_chunk_len must be at least 1")
        if max_chunk_len <= min_chunk_len:
            raise ValueError("max_chunk_len must be greater than min_chunk_len")
        if is_given(speed):
            _validate_speed(speed)

        # Use provided tokenizer or create default
        resolved_word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
        if is_given(word_tokenizer):
            resolved_word_tokenizer = word_tokenizer
        else:
            resolved_word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            voice_settings=VoiceSettings(
                voice_id=voice_id,
                output_format=output_format,
                speed=speed if is_given(speed) else None,
            ),
            word_tokenizer=resolved_word_tokenizer,
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=num_channels,
            phrase_replacement_config_id=phrase_replacement_config_id
            if is_given(phrase_replacement_config_id)
            else None,
            min_chunk_len=min_chunk_len,
            max_chunk_len=max_chunk_len,
        )

        self._client: WebSocketClient | None = None
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._prewarm_task: asyncio.Task[None] | None = None

        logger.info(
            "UpliftAI TTS initialized (sentence-chunked streaming): "
            f"min_chunk_len={min_chunk_len} max_chunk_len={max_chunk_len} "
            f"pipelined={MAX_PIPELINED_REQUESTS} output_format={output_format}"
        )

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[OutputFormat] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS configuration options.

        Args:
            voice_id: New voice ID
            output_format: New output format (see __init__ for options)
            speed: New speaking rate, 0.5 to 2.0 (see __init__)
        """
        if is_given(voice_id):
            self._opts.voice_settings.voice_id = voice_id
        if is_given(output_format):
            self._opts.voice_settings.output_format = output_format
        if is_given(speed):
            _validate_speed(speed)
            self._opts.voice_settings.speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text to speech using chunked stream."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming synthesis session."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Initiate the WebSocket connection to the TTS service without blocking.

        This starts a background task that connects if not already connected, so the
        first synthesis request doesn't pay the connection cost.
        """
        if self._prewarm_task is not None:
            return
        if self._client and self._client.connected:
            return

        if not self._client:
            self._client = WebSocketClient(self._opts)
        client = self._client

        async def _prewarm_impl() -> None:
            if not await client.connect():
                logger.warning("failed to prewarm TTS WebSocket connection")

        # hold a strong reference: the event loop alone won't keep the task alive
        task = asyncio.create_task(_prewarm_impl())
        task.add_done_callback(lambda _: setattr(self, "_prewarm_task", None))
        self._prewarm_task = task

    async def aclose(self) -> None:
        """Clean up resources"""
        if self._prewarm_task is not None:
            await utils.aio.gracefully_cancel(self._prewarm_task)
            self._prewarm_task = None

        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        if self._client:
            await self._client.disconnect()
            self._client = None


class WebSocketClient:
    """Manages WebSocket connection to TTS service"""

    def __init__(self, opts: _TTSOptions):
        self.opts = opts
        self.sio: socketio.AsyncClient | None = None
        self.connected = False
        # each queue yields audio bytes, an Exception on failure, or None when done
        self.audio_callbacks: dict[str, asyncio.Queue[bytes | Exception | None]] = {}
        self.active_requests: dict[str, bool] = {}
        # serializes concurrent connect() calls (e.g. prewarm racing the first synthesis)
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        async with self._connect_lock:
            if self.connected:
                return True

            try:
                # drop any previous socket before replacing it: a stale client
                # left auto-reconnecting in the background shares our handlers
                # and could flip self.connected under a new connection
                if self.sio is not None:
                    try:
                        await self.sio.disconnect()
                    except Exception:
                        pass
                    self.sio = None

                # reconnection is handled by the SDK retry loop (which replays
                # the turn), not by socket.io auto-reconnect
                self.sio = socketio.AsyncClient(
                    reconnection=False,
                    logger=False,
                    engineio_logger=False,
                )

                # Register handlers
                self.sio.on("message", self._on_message, namespace=WEBSOCKET_NAMESPACE)
                self.sio.on("connect", self._on_connect, namespace=WEBSOCKET_NAMESPACE)
                self.sio.on("disconnect", self._on_disconnect, namespace=WEBSOCKET_NAMESPACE)

                # Prepare auth
                auth_data = {"token": self.opts.api_key}

                # Connect
                await self.sio.connect(
                    self.opts.base_url,
                    auth=auth_data,
                    namespaces=[WEBSOCKET_NAMESPACE],
                    transports=["websocket"],
                    wait_timeout=10,
                )

                # Wait for connection
                max_wait = 5.0
                start_time = time.time()
                while not self.connected and (time.time() - start_time) < max_wait:
                    await asyncio.sleep(0.1)

                if not self.connected and self.sio.connected:
                    self.connected = True

                return self.connected

            except Exception as e:
                logger.error(f"Connection failed: {e}")
                return False

    async def synthesize(
        self, text: str, request_id: str | None = None
    ) -> asyncio.Queue[bytes | Exception | None]:
        """Send synthesis request and return audio queue"""
        if not self.sio or not self.connected:
            if not await self.connect():
                raise APIConnectionError("Failed to connect to TTS service")

        if not request_id:
            request_id = str(uuid.uuid4())

        # Create audio queue
        audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        self.audio_callbacks[request_id] = audio_queue
        self.active_requests[request_id] = True

        # Build message
        message: dict[str, Any] = {
            "type": "synthesize",
            "requestId": request_id,
            "text": text,
            "voiceId": self.opts.voice_settings.voice_id,
            "outputFormat": self.opts.voice_settings.output_format,
        }

        if self.opts.phrase_replacement_config_id:
            message["phraseReplacementConfigId"] = self.opts.phrase_replacement_config_id

        if self.opts.voice_settings.speed is not None:
            message["speed"] = self.opts.voice_settings.speed

        logger.debug(
            f"Sending synthesis request {request_id[:8]}", extra={"lk.pii.text": text[:50]}
        )

        try:
            if self.sio is not None:
                await self.sio.emit("synthesize", message, namespace=WEBSOCKET_NAMESPACE)
        except Exception as e:
            logger.error(f"Failed to emit synthesis: {e}")
            del self.audio_callbacks[request_id]
            del self.active_requests[request_id]
            raise

        return audio_queue

    async def cancel(self, request_id: str) -> None:
        """Cancel an active synthesis request and drop any further audio for it"""
        if request_id in self.audio_callbacks:
            await self.audio_callbacks[request_id].put(None)
            del self.audio_callbacks[request_id]
        self.active_requests.pop(request_id, None)

        if self.sio is None or not self.connected:
            return

        try:
            await self.sio.emit(
                "cancel",
                {"type": "cancel", "requestId": request_id},
                namespace=WEBSOCKET_NAMESPACE,
            )
        except Exception as e:
            logger.warning(f"Failed to send cancel for request {request_id[:8]}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from service"""
        # close unconditionally, not only when the ready handshake completed: a
        # connect() cancelled mid-handshake leaves an open socket with
        # self.connected still False, which must not survive shutdown
        self.connected = False
        if self.sio is not None:
            try:
                await self.sio.disconnect()
            except Exception as e:
                logger.warning(f"Error closing TTS WebSocket: {e}")
            self.sio = None

    async def _on_connect(self) -> None:
        """Handle connection"""
        logger.debug("WebSocket connected")

    async def _on_message(self, data: Any) -> None:
        """Handle messages"""
        message_type = data.get("type")

        if message_type == "ready":
            self.connected = True
            logger.debug(f"Ready with session: {data.get('sessionId')}")

        elif message_type == "audio":
            request_id = data.get("requestId")
            audio_b64 = data.get("audio")

            if audio_b64 and request_id in self.audio_callbacks:
                audio_bytes = base64.b64decode(audio_b64)
                if self.active_requests.get(request_id, False):
                    await self.audio_callbacks[request_id].put(audio_bytes)

        elif message_type == "audio_end":
            request_id = data.get("requestId")
            if request_id in self.audio_callbacks:
                await self.audio_callbacks[request_id].put(None)
                del self.audio_callbacks[request_id]
                if request_id in self.active_requests:
                    del self.active_requests[request_id]

        elif message_type == "error":
            request_id = data.get("requestId", "unknown")
            error_msg = data.get("message", str(data))
            logger.error(f"Error for {request_id}: {error_msg}")

            if request_id in self.audio_callbacks:
                await self.audio_callbacks[request_id].put(APIError(f"TTS error: {error_msg}"))
                del self.audio_callbacks[request_id]
                if request_id in self.active_requests:
                    del self.active_requests[request_id]

    async def _on_disconnect(self) -> None:
        """Handle disconnection"""
        self.connected = False
        for queue in self.audio_callbacks.values():
            await queue.put(APIConnectionError("connection to TTS service closed"))
        self.audio_callbacks.clear()
        self.active_requests.clear()


class ChunkedStream(tts.ChunkedStream):
    """Chunked synthesis implementation"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute synthesis"""
        request_id = utils.shortuuid()

        try:
            # Initialize emitter
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._tts._opts.sample_rate,
                num_channels=self._tts._opts.num_channels,
                mime_type=get_content_type_from_output_format(
                    self._tts._opts.voice_settings.output_format
                ),
            )

            # Create client if needed
            if not self._tts._client:
                self._tts._client = WebSocketClient(self._tts._opts)
            client = self._tts._client

            # Get audio queue
            audio_queue = await client.synthesize(self._input_text, request_id)

            # Stream audio
            try:
                while True:
                    try:
                        audio_data = await asyncio.wait_for(
                            audio_queue.get(), timeout=AUDIO_CHUNK_TIMEOUT
                        )
                    except asyncio.TimeoutError as e:
                        raise APITimeoutError("timed out waiting for TTS audio") from e

                    if audio_data is None:
                        break
                    if isinstance(audio_data, Exception):
                        raise audio_data

                    output_emitter.push(audio_data)
            except BaseException:
                # on interruption, timeout or error, stop the server-side synthesis
                await client.cancel(request_id)
                raise

            output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(f"TTS synthesis failed: {str(e)}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis implementation.

    Buffered text is flushed to the TTS service at sentence boundaries (or after
    ``max_chunk_len`` characters), and up to ``MAX_PIPELINED_REQUESTS`` requests are
    kept in flight while earlier audio is still being emitted.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute streaming synthesis"""
        request_id = utils.shortuuid()
        # local so a retry attempt starts with a fresh channel instead of one
        # already closed (or holding stale token streams) from a prior attempt
        segments_ch = utils.aio.Chan[tokenize.WordStream | tokenize.SentenceStream]()

        # chunks are decoded in-plugin (see _emit_chunk), the emitter gets raw PCM
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._opts.sample_rate,
            num_channels=self._tts._opts.num_channels,
            stream=True,
            mime_type="audio/pcm",
        )

        async def _tokenize_input() -> None:
            """Tokenize input text"""
            token_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if token_stream is None:
                        token_stream = self._tts._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(token_stream)

                    token_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if token_stream is not None:
                        token_stream.end_input()
                    token_stream = None

            if token_stream is not None:
                token_stream.end_input()

            segments_ch.close()

        async def _process_segments() -> None:
            """Process segments"""
            async for token_stream in segments_ch:
                await self._run_segment(token_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(f"TTS stream failed: {str(e)}") from e
        finally:
            # emitter finalization is the base class's job: end_input() on success,
            # aclose() on failure (which discards any partial tail frame)
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_segment(
        self,
        token_stream: tokenize.WordStream | tokenize.SentenceStream,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        """Process a single segment"""
        opts = self._tts._opts

        if not self._tts._client:
            self._tts._client = WebSocketClient(opts)
        client = self._tts._client

        output_emitter.start_segment(segment_id=utils.shortuuid())

        # audio queues of in-flight requests, in submission order. the bound caps
        # look-ahead at MAX_PIPELINED_REQUESTS queued, plus one being emitted and
        # one blocked in _submit_buffer
        inflight_ch: asyncio.Queue[tuple[str, asyncio.Queue[bytes | Exception | None]] | None] = (
            asyncio.Queue(maxsize=MAX_PIPELINED_REQUESTS)
        )
        # requests submitted but not fully drained; cancelled server-side on teardown
        pending_request_ids: set[str] = set()

        async def _schedule_chunks() -> None:
            buffer: list[str] = []
            buffer_len = 0

            async def _submit_buffer() -> None:
                nonlocal buffer, buffer_len
                if not buffer:
                    return

                if isinstance(opts.word_tokenizer, tokenize.WordTokenizer):
                    chunk_text = opts.word_tokenizer.format_words(buffer)
                else:
                    chunk_text = " ".join(buffer)
                buffer, buffer_len = [], 0

                if not chunk_text.strip():
                    return

                self._mark_started()
                chunk_request_id = str(uuid.uuid4())
                # track before the await: if we're cancelled mid-submission, the
                # request may already be on the wire and must still get cancelled
                pending_request_ids.add(chunk_request_id)
                audio_queue = await client.synthesize(chunk_text, chunk_request_id)
                await inflight_ch.put((chunk_request_id, audio_queue))

            async for token_data in token_stream:
                token = token_data.token
                if buffer and buffer_len + len(token) + 1 > opts.max_chunk_len:
                    await _submit_buffer()

                buffer.append(token)
                buffer_len += len(token) + 1

                if buffer_len >= opts.min_chunk_len and _ends_sentence(token):
                    await _submit_buffer()

            await _submit_buffer()
            await inflight_ch.put(None)

        async def _iter_audio(
            audio_queue: asyncio.Queue[bytes | Exception | None],
        ) -> AsyncIterator[bytes]:
            while True:
                try:
                    audio_data = await asyncio.wait_for(
                        audio_queue.get(), timeout=AUDIO_CHUNK_TIMEOUT
                    )
                except asyncio.TimeoutError as e:
                    raise APITimeoutError("timed out waiting for TTS audio") from e

                if audio_data is None:
                    return
                if isinstance(audio_data, Exception):
                    raise audio_data

                yield audio_data

        async def _emit_chunk(audio_queue: asyncio.Queue[bytes | Exception | None]) -> None:
            # each request returns a complete audio file. a single decoder cannot
            # span multiple files (a mid-stream MP3/RIFF header corrupts or
            # truncates decoding), so every chunk is decoded independently and
            # the emitter only ever sees raw PCM
            if opts.voice_settings.output_format == "PCM_22050_16":
                async for audio_data in _iter_audio(audio_queue):
                    output_emitter.push(audio_data)
                return

            decoder = codecs.AudioStreamDecoder(
                sample_rate=opts.sample_rate,
                num_channels=opts.num_channels,
                format=get_content_type_from_output_format(opts.voice_settings.output_format),
            )
            fed_bytes = 0
            decoded_frames = 0

            async def _feed_decoder() -> None:
                nonlocal fed_bytes
                try:
                    async for audio_data in _iter_audio(audio_queue):
                        fed_bytes += len(audio_data)
                        decoder.push(audio_data)
                finally:
                    decoder.end_input()

            feed_task = asyncio.create_task(_feed_decoder())
            try:
                async for frame in decoder:
                    decoded_frames += 1
                    output_emitter.push(frame.data.tobytes())
                await feed_task  # propagate synthesis errors (APIError, timeout)
            finally:
                await utils.aio.gracefully_cancel(feed_task)
                await decoder.aclose()

            # the SDK decoder is fail-open: a decode error only logs and closes the
            # stream. surface it as a retryable error instead of a silent gap in speech
            if fed_bytes > 0 and decoded_frames == 0:
                raise APIError(f"TTS audio chunk could not be decoded ({fed_bytes} bytes received)")

        async def _emit_audio() -> None:
            while (inflight := await inflight_ch.get()) is not None:
                chunk_request_id, audio_queue = inflight
                # on failure the id stays in pending_request_ids so teardown sends
                # a cancel — required for timeouts, harmless for dead requests
                await _emit_chunk(audio_queue)
                pending_request_ids.discard(chunk_request_id)
                # release the held-back tail frame so a chunk's final audio isn't
                # delayed until the next chunk's audio arrives
                output_emitter.flush()

        tasks = [
            asyncio.create_task(_schedule_chunks()),
            asyncio.create_task(_emit_audio()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            # on interruption or error, stop synthesis of requests we'll never play
            for chunk_request_id in pending_request_ids:
                await client.cancel(chunk_request_id)

        output_emitter.end_segment()
