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
from dataclasses import dataclass
from typing import Any, Literal, cast

import socketio  # type: ignore[import-not-found]

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
from livekit.agents.utils import is_given

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


@dataclass
class VoiceSettings:
    """Voice configuration settings"""

    voice_id: str = DEFAULT_VOICE_ID
    output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT


@dataclass
class _TTSOptions:
    """Internal TTS options"""

    base_url: str
    api_key: str
    voice_settings: VoiceSettings
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    sample_rate: int
    num_channels: int


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
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Uplift TTS.

        Args:
            base_url: Base URL for TTS service. Defaults to wss://api.upliftai.org
            api_key: API key for authentication
            voice_id: Voice ID to use. Defaults to "17"
            output_format: Audio output format. Options:
                - 'PCM_22050_16': PCM format, 22.05kHz, 16-bit
                - 'WAV_22050_16': WAV format, 22.05kHz, 16-bit
                - 'WAV_22050_32': WAV format, 22.05kHz, 32-bit
                - 'MP3_22050_32': MP3 format, 22.05kHz, 32kbps (default)
                - 'MP3_22050_64': MP3 format, 22.05kHz, 64kbps
                - 'MP3_22050_128': MP3 format, 22.05kHz, 128kbps
                - 'OGG_22050_16': OGG format, 22.05kHz, 16-bit
                - 'ULAW_8000_8': μ-law format, 8kHz, 8-bit
            sample_rate: Sample rate for audio output. Defaults to 22050
            num_channels: Number of audio channels. Defaults to 1 (mono)
            word_tokenizer: Tokenizer for processing text. Defaults to `livekit.agents.tokenize.basic.WordTokenizer`.
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

        # Use provided tokenizer or create default
        resolved_word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
        if is_given(word_tokenizer):
            resolved_word_tokenizer = cast(
                tokenize.WordTokenizer | tokenize.SentenceTokenizer, word_tokenizer
            )
        else:
            resolved_word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            voice_settings=VoiceSettings(voice_id=voice_id, output_format=output_format),
            word_tokenizer=resolved_word_tokenizer,
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=num_channels,
        )

        self._client: WebSocketClient | None = None
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[OutputFormat] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS configuration options.

        Args:
            voice_id: New voice ID
            output_format: New output format (see __init__ for options)
        """
        if is_given(voice_id):
            self._opts.voice_settings.voice_id = voice_id
        if is_given(output_format):
            self._opts.voice_settings.output_format = cast(OutputFormat, output_format)

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

    async def aclose(self) -> None:
        """Clean up resources"""
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
        self.audio_callbacks: dict[str, asyncio.Queue[bytes | None]] = {}
        self.active_requests: dict[str, bool] = {}

    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        if self.connected:
            return True

        try:
            self.sio = socketio.AsyncClient(
                reconnection=True,
                reconnection_attempts=3,
                reconnection_delay=1,
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
    ) -> asyncio.Queue[bytes | None]:
        """Send synthesis request and return audio queue"""
        if not self.sio or not self.connected:
            if not await self.connect():
                raise ConnectionError("Failed to connect to TTS service")

        if not request_id:
            request_id = str(uuid.uuid4())

        # Create audio queue
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.audio_callbacks[request_id] = audio_queue
        self.active_requests[request_id] = True

        # Build message
        message = {
            "type": "synthesize",
            "requestId": request_id,
            "text": text,
            "voiceId": self.opts.voice_settings.voice_id,
            "outputFormat": self.opts.voice_settings.output_format,
        }

        logger.debug(f"Sending synthesis request {request_id[:8]} for text: '{text[:50]}...'")

        try:
            if self.sio is not None:
                await self.sio.emit("synthesize", message, namespace=WEBSOCKET_NAMESPACE)
        except Exception as e:
            logger.error(f"Failed to emit synthesis: {e}")
            del self.audio_callbacks[request_id]
            del self.active_requests[request_id]
            raise

        return audio_queue

    async def disconnect(self) -> None:
        """Disconnect from service"""
        if self.sio and self.connected:
            await self.sio.disconnect()
            self.connected = False

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
                await self.audio_callbacks[request_id].put(None)
                del self.audio_callbacks[request_id]
                if request_id in self.active_requests:
                    del self.active_requests[request_id]

    async def _on_disconnect(self) -> None:
        """Handle disconnection"""
        self.connected = False
        for queue in self.audio_callbacks.values():
            await queue.put(None)
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

            # Get audio queue
            audio_queue = await self._tts._client.synthesize(self._input_text, request_id)

            # Stream audio
            while True:
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=30.0)

                    if audio_data is None:
                        break

                    output_emitter.push(audio_data)

                except asyncio.TimeoutError:
                    logger.warning("Audio timeout")
                    break

            output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError(f"TTS synthesis failed: {str(e)}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis implementation"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute streaming synthesis"""
        request_id = utils.shortuuid()
        segments_ch = utils.aio.Chan[tokenize.WordStream | tokenize.SentenceStream]()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._opts.sample_rate,
            num_channels=self._tts._opts.num_channels,
            stream=True,
            mime_type=get_content_type_from_output_format(
                self._tts._opts.voice_settings.output_format
            ),
        )

        async def _tokenize_input() -> None:
            """Tokenize input text"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._tts._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(word_stream)

                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream is not None:
                        word_stream.end_input()
                    word_stream = None

            if word_stream is not None:
                word_stream.end_input()

            segments_ch.close()

        async def _process_segments() -> None:
            """Process segments"""
            async for word_stream in segments_ch:
                await self._run_segment(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_segment(
        self,
        word_stream: tokenize.WordStream | tokenize.SentenceStream,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        """Process a single segment"""
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        try:
            # Create client if needed
            if not self._tts._client:
                self._tts._client = WebSocketClient(self._tts._opts)

            # Collect text
            text_parts = []
            async for data in word_stream:
                text_parts.append(data.token)

            if not text_parts:
                output_emitter.end_segment()
                return

            # Format text
            if isinstance(self._tts._opts.word_tokenizer, tokenize.WordTokenizer):
                full_text = self._tts._opts.word_tokenizer.format_words(text_parts)
            else:
                full_text = " ".join(text_parts)

            self._mark_started()

            # Synthesize
            request_id = str(uuid.uuid4())
            audio_queue = await self._tts._client.synthesize(full_text, request_id)

            # Stream audio
            while True:
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=30.0)

                    if audio_data is None:
                        break

                    output_emitter.push(audio_data)

                except asyncio.TimeoutError:
                    break

            output_emitter.end_segment()

        except Exception as e:
            logger.error(f"Segment synthesis error: {e}")
            raise APIError(f"Segment synthesis failed: {str(e)}") from e
