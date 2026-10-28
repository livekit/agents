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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol, cast

from pydantic import BaseModel, JsonValue

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
    phrase_replacement_config_id: str | None


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
            phrase_replacement_config_id: Optional ID for phrase replacement configuration
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
            resolved_word_tokenizer = word_tokenizer
        else:
            resolved_word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            voice_settings=VoiceSettings(voice_id=voice_id, output_format=output_format),
            word_tokenizer=resolved_word_tokenizer,
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=num_channels,
            phrase_replacement_config_id=phrase_replacement_config_id
            if is_given(phrase_replacement_config_id)
            else None,
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
            self._opts.voice_settings.output_format = output_format

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text to speech using chunked stream."""
        return ChunkedStream(
            tts=self,
            opts=self._opts,
            synthesize_request=self._synthesize_request,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming synthesis session."""
        stream = SynthesizeStream(
            tts=self,
            opts=self._opts,
            synthesize_request=self._synthesize_request,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def _synthesize_request(
        self, text: str, request_id: str
    ) -> asyncio.Queue[bytes | APIError | None]:
        if self._client is None:
            self._client = WebSocketClient(self._opts)
        return await self._client.synthesize(text, request_id)

    async def aclose(self) -> None:
        """Clean up resources"""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

        if self._client:
            await self._client.disconnect()
            self._client = None


class _SocketClient(Protocol):
    connected: bool

    def on(
        self,
        event: str,
        handler: Callable[[], Awaitable[None]] | Callable[[object], Awaitable[None]],
        *,
        namespace: str,
    ) -> None: ...
    async def connect(
        self,
        url: str,
        *,
        auth: dict[str, str],
        namespaces: list[str],
        transports: list[str],
        wait_timeout: float,
    ) -> None: ...
    async def emit(self, event: str, data: dict[str, JsonValue], *, namespace: str) -> None: ...
    async def disconnect(self) -> None: ...


class _SocketFactory(Protocol):
    def __call__(
        self,
        *,
        reconnection: bool,
        logger: bool,
        engineio_logger: bool,
    ) -> _SocketClient: ...


class _ServerMessage(BaseModel):
    type: str
    requestId: str | None = None
    sessionId: str | None = None
    audio: str | None = None
    message: str | None = None


class WebSocketClient:
    """Manages WebSocket connection to TTS service"""

    def __init__(self, opts: _TTSOptions):
        self.opts = opts
        self.sio: _SocketClient | None = None
        self.connected = False
        self.audio_callbacks: dict[str, asyncio.Queue[bytes | APIError | None]] = {}
        self.active_requests: dict[str, bool] = {}

    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        if self.connected:
            return True

        try:
            # python-socketio does not ship a complete typed AsyncClient contract.
            factory = cast(_SocketFactory, import_module("socketio").AsyncClient)
            self.sio = factory(
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
    ) -> asyncio.Queue[bytes | APIError | None]:
        """Send synthesis request and return audio queue"""
        if not self.sio or not self.connected:
            if not await self.connect():
                raise ConnectionError("Failed to connect to TTS service")

        if not request_id:
            request_id = str(uuid.uuid4())

        # Create audio queue
        audio_queue: asyncio.Queue[bytes | APIError | None] = asyncio.Queue()
        self.audio_callbacks[request_id] = audio_queue
        self.active_requests[request_id] = True

        # Build message
        message: dict[str, JsonValue] = {
            "type": "synthesize",
            "requestId": request_id,
            "text": text,
            "voiceId": self.opts.voice_settings.voice_id,
            "outputFormat": self.opts.voice_settings.output_format,
        }

        if self.opts.phrase_replacement_config_id:
            message["phraseReplacementConfigId"] = self.opts.phrase_replacement_config_id

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

    async def disconnect(self) -> None:
        """Disconnect from service"""
        if self.sio and self.connected:
            await self.sio.disconnect()
            self.connected = False

    async def _on_connect(self) -> None:
        """Handle connection"""
        logger.debug("WebSocket connected")

    async def _on_message(self, data: object) -> None:
        message = _ServerMessage.model_validate(data)
        if message.type == "ready":
            self.connected = True
            logger.debug("Ready with session: %s", message.sessionId)
            return
        request_id = message.requestId
        if request_id is None:
            return
        queue = self.audio_callbacks.get(request_id)
        if message.type == "audio":
            if message.audio and queue is not None and self.active_requests.get(request_id, False):
                await queue.put(base64.b64decode(message.audio))
        elif message.type == "audio_end":
            if queue is not None:
                await queue.put(None)
                self.audio_callbacks.pop(request_id, None)
                self.active_requests.pop(request_id, None)
        elif message.type == "error" and queue is not None:
            await queue.put(
                APIError(
                    "Uplift rejected the synthesis request",
                    body=message.model_dump(exclude_none=True),
                    retryable=False,
                )
            )
            self.audio_callbacks.pop(request_id, None)
            self.active_requests.pop(request_id, None)

    async def _on_disconnect(self) -> None:
        self.connected = False
        for queue in self.audio_callbacks.values():
            await queue.put(APIConnectionError("Uplift disconnected before synthesis completed"))
        self.audio_callbacks.clear()
        self.active_requests.clear()


class ChunkedStream(tts.ChunkedStream):
    """Chunked synthesis implementation"""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        synthesize_request: Callable[[str, str], Awaitable[asyncio.Queue[bytes | APIError | None]]],
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._synthesize_request = synthesize_request

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute synthesis"""
        request_id = utils.shortuuid()

        try:
            # Initialize emitter
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                mime_type=get_content_type_from_output_format(
                    self._opts.voice_settings.output_format
                ),
            )

            audio_queue = await self._synthesize_request(self._input_text, request_id)

            # Stream audio
            while True:
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=30.0)

                    if audio_data is None:
                        break
                    if isinstance(audio_data, APIError):
                        raise audio_data

                    output_emitter.push(audio_data)

                except asyncio.TimeoutError as exc:
                    raise APITimeoutError("Uplift did not complete synthesis") from exc

            output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError("Uplift synthesis failed") from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis implementation"""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        synthesize_request: Callable[[str, str], Awaitable[asyncio.Queue[bytes | APIError | None]]],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._synthesize_request = synthesize_request

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Execute streaming synthesis"""
        request_id = utils.shortuuid()
        segments_ch = utils.aio.Chan[tokenize.WordStream | tokenize.SentenceStream]()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            stream=True,
            mime_type=get_content_type_from_output_format(self._opts.voice_settings.output_format),
        )

        async def _tokenize_input() -> None:
            """Tokenize input text"""
            word_stream = None
            async for input in self.input:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(word_stream)

                    word_stream.push_text(input)
                else:
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
        except APIError:
            raise
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
            # Collect text
            text_parts: list[str] = []
            async for data in word_stream:
                text_parts.append(data.token)

            if not text_parts:
                return

            # Format text
            if isinstance(self._opts.word_tokenizer, tokenize.WordTokenizer):
                full_text = self._opts.word_tokenizer.format_words(text_parts)
            else:
                full_text = " ".join(text_parts)

            self.mark_started()

            # Synthesize
            audio_queue = await self._synthesize_request(full_text, segment_id)

            # Stream audio
            while True:
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=30.0)

                    if audio_data is None:
                        break
                    if isinstance(audio_data, APIError):
                        raise audio_data

                    output_emitter.push(audio_data)

                except asyncio.TimeoutError as exc:
                    raise APITimeoutError("Uplift did not complete synthesis") from exc

            output_emitter.end_input()

        except APIError:
            raise
        except Exception as e:
            raise APIError("Uplift segment synthesis failed") from e
