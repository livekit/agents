from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

from fish_audio_sdk import (  # type: ignore[import-untyped]
    AsyncWebSocketSession,
    Session as FishAudioSession,
    TTSRequest,
)
from fish_audio_sdk.exceptions import WebSocketErr  # type: ignore[import-untyped]
from fish_audio_sdk.schemas import Backends  # type: ignore[import-untyped]

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import LatencyMode, OutputFormat

DEFAULT_MODEL: Backends = "s1"
DEFAULT_REFERENCE_ID = "8ef4a238714b45718ce04243307c57a7"


@dataclass
class _TTSOptions:
    model: Backends
    output_format: OutputFormat
    sample_rate: int
    num_channels: int
    reference_id: NotGivenOr[str]
    base_url: str
    api_key: str
    latency_mode: LatencyMode


class TTS(tts.TTS):
    """
    Fish Audio TTS implementation for LiveKit Agents.

    This plugin provides text-to-speech synthesis using Fish Audio's API.
    It supports both chunked (non-streaming) and real-time WebSocket streaming modes,
    as well as reference ID-based and custom reference audio-based synthesis.

    Args:
        api_key (NotGivenOr[str]): Fish Audio API key. Can be set via argument or `FISH_API_KEY` environment variable.
        model (Backends): TTS model/backend to use. Defaults to "s1".
        reference_id (NotGivenOr[str]): Reference voice model ID. Defaults to a general-purpose voice.
        output_format (OutputFormat): Audio output format. Defaults to "pcm" for streaming.
        sample_rate (int): Audio sample rate in Hz. Defaults to 24000.
        num_channels (int): Number of audio channels. Defaults to 1 (mono).
        base_url (NotGivenOr[str]): Custom base URL for the Fish Audio API. Optional.
        latency_mode (LatencyMode): Streaming latency mode for WebSocket. "normal" (~500ms) or "balanced" (~300ms). Defaults to "balanced".
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: Backends = DEFAULT_MODEL,
        reference_id: NotGivenOr[str] = DEFAULT_REFERENCE_ID,
        output_format: OutputFormat = "pcm",
        sample_rate: int = 24000,
        num_channels: int = 1,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        latency_mode: LatencyMode = "balanced",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        fish_api_key = api_key if is_given(api_key) else os.getenv("FISH_API_KEY")
        if not fish_api_key:
            raise ValueError(
                "Fish Audio API key is required, either as argument or set FISH_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            model=model,
            output_format=output_format,
            sample_rate=sample_rate,
            num_channels=num_channels,
            reference_id=reference_id,
            base_url=base_url if is_given(base_url) else "https://api.fish.audio",
            api_key=fish_api_key,
            latency_mode=latency_mode,
        )

        # Initialize Fish Audio sessions
        self._session = FishAudioSession(self._opts.api_key, base_url=self._opts.base_url)

        # WebSocket session for streaming (lazy initialized)
        self._ws_session: AsyncWebSocketSession | None = None
        self._ws_session_lock = asyncio.Lock()

        # Track active streams
        self._streams = weakref.WeakSet[SynthesizeStream]()

        logger.info(
            "FishAudioTTS initialized",
            extra={
                "model": self._opts.model,
                "format": self._opts.output_format,
                "sample_rate": self._opts.sample_rate,
                "latency_mode": self._opts.latency_mode,
            },
        )

    @property
    def model(self) -> Backends:
        return self._opts.model

    @property
    def output_format(self) -> OutputFormat:
        return self._opts.output_format

    @property
    def reference_id(self) -> NotGivenOr[str]:
        return self._opts.reference_id

    @property
    def session(self) -> FishAudioSession:
        return self._session

    @property
    def latency_mode(self) -> LatencyMode:
        return self._opts.latency_mode

    async def _ensure_ws_session(self) -> AsyncWebSocketSession:
        """Get the current WebSocket session, creating one if needed"""
        async with self._ws_session_lock:
            if self._ws_session is None:
                self._ws_session = AsyncWebSocketSession(
                    apikey=self._opts.api_key, base_url=self._opts.base_url
                )
            return self._ws_session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize speech from text using chunked (non-streaming) mode.

        Args:
            text (str): The text to synthesize.
            conn_options (APIConnectOptions): Connection options for the API call.

        Returns:
            ChunkedStream: A stream object that will produce synthesized audio.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """
        Create a real-time streaming TTS session using WebSocket.

        Args:
            conn_options (APIConnectOptions): Connection options for the WebSocket.

        Returns:
            SynthesizeStream: A streaming object for real-time text-to-speech.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """
        Close TTS resources and WebSocket sessions.
        """
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self._ws_session is not None:
            await self._ws_session.close()
            self._ws_session = None


class ChunkedStream(tts.ChunkedStream):
    """
    ChunkedStream implementation for Fish Audio TTS.

    This class handles non-streaming synthesis by communicating with the Fish Audio
    REST API and returning complete audio data.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Generate audio and push it to the emitter.

        Args:
            output_emitter (tts.AudioEmitter): The emitter to receive synthesized audio data.

        Raises:
            APIConnectionError: If connection to Fish Audio fails.
            APITimeoutError: If the request times out.
            APIStatusError: If Fish Audio API returns an error.
        """
        try:
            audio_data = await asyncio.wait_for(
                asyncio.to_thread(self._generate_audio_sync), timeout=self._conn_options.timeout
            )

            if not audio_data:
                logger.warning("No audio data generated from Fish Audio API")
                return

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                mime_type=f"audio/{self._opts.output_format}",
            )
            output_emitter.push(audio_data)
            output_emitter.flush()

        except asyncio.TimeoutError as e:
            logger.error(
                "Fish Audio TTS request timed out",
                extra={
                    "timeout": self._conn_options.timeout,
                    "text_length": len(self._input_text),
                },
            )
            raise APITimeoutError(
                f"Fish Audio TTS request timed out after {self._conn_options.timeout}s"
            ) from e
        except APITimeoutError:
            # Already logged - re-raise without wrapping
            raise
        except (APIConnectionError, APIStatusError):
            # Already logged - re-raise without wrapping
            raise
        except Exception as e:
            # Unexpected errors only (shouldn't happen in normal operation)
            logger.error(
                "Unexpected error in Fish Audio TTS synthesis",
                exc_info=e,
                extra={"text_length": len(self._input_text)},
            )
            raise APIConnectionError("Fish Audio TTS synthesis failed") from e

    def _generate_audio_sync(self) -> bytes:
        """
        Synchronously generate audio using Fish Audio SDK.

        Returns:
            bytes: The synthesized audio data.

        Raises:
            APIConnectionError: If the SDK call fails.
        """
        request = TTSRequest(
            text=self._input_text,
            reference_id=self._opts.reference_id if is_given(self._opts.reference_id) else None,
            format=self._opts.output_format,
            sample_rate=self._opts.sample_rate,
        )

        audio_data = bytearray()
        try:
            for chunk in self._tts.session.tts(request, backend=self._opts.model):
                audio_data.extend(chunk)
        except Exception as e:
            logger.error(
                "Fish Audio SDK TTS call failed",
                exc_info=e,
                extra={
                    "text_length": len(self._input_text),
                    "reference_id": self._opts.reference_id,
                    "format": self._opts.output_format,
                },
            )
            raise APIConnectionError("Fish Audio SDK TTS call failed") from e

        return bytes(audio_data)


class SynthesizeStream(tts.SynthesizeStream):
    """
    Real-time streaming TTS implementation using WebSocket.

    This class handles incremental text input and streams audio output in real-time,
    optimized for low-latency interactive applications like chatbots and voice assistants.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._request_id = utils.shortuuid()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Run the streaming TTS session.

        Args:
            output_emitter (tts.AudioEmitter): The emitter to receive audio chunks.

        Raises:
            APIConnectionError: If WebSocket connection fails.
            APITimeoutError: If WebSocket connection times out.
            APIStatusError: If Fish Audio returns an error.
        """
        output_emitter.initialize(
            request_id=self._request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            stream=True,
            mime_type=f"audio/{self._opts.output_format}",
        )
        output_emitter.start_segment(segment_id=self._request_id)

        try:
            ws_session = await asyncio.wait_for(
                self._tts._ensure_ws_session(), timeout=self._conn_options.timeout
            )

            await self._stream_audio(ws_session, output_emitter)

        except asyncio.TimeoutError as e:
            logger.error(
                "Failed to connect to Fish Audio WebSocket",
                extra={
                    "timeout": self._conn_options.timeout,
                    "latency_mode": self._opts.latency_mode,
                },
            )
            raise APITimeoutError(
                f"Fish Audio WebSocket connection timed out after {self._conn_options.timeout}s"
            ) from e
        except APITimeoutError:
            # Already logged - re-raise without wrapping
            raise
        except (APIConnectionError, APIStatusError):
            # Already logged - re-raise without wrapping
            raise
        except Exception as e:
            # Unexpected errors
            logger.error(
                "Unexpected error during Fish Audio WebSocket streaming",
                exc_info=e,
                extra={"latency_mode": self._opts.latency_mode},
            )
            raise APIStatusError(f"Fish Audio streaming failed: {e}") from e
        finally:
            output_emitter.end_segment()

    async def _stream_audio(
        self, ws_session: AsyncWebSocketSession, output_emitter: tts.AudioEmitter
    ) -> None:
        """
        Internal method to handle the actual WebSocket streaming.

        Args:
            ws_session (AsyncWebSocketSession): Connected WebSocket session.
            output_emitter (tts.AudioEmitter): The emitter to receive audio chunks.

        Raises:
            APIConnectionError: If WebSocket connection fails.
            WebSocketErr: If Fish Audio WebSocket returns an error.
        """

        # Create TTS request for streaming
        request = TTSRequest(
            text="",  # Empty for streaming mode
            reference_id=self._opts.reference_id if is_given(self._opts.reference_id) else None,
            format=self._opts.output_format,
            sample_rate=self._opts.sample_rate,
            latency=self._opts.latency_mode,
        )

        async def text_generator() -> AsyncIterator[str]:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                yield data

        try:
            audio_iterator = ws_session.tts(
                request=request, text_stream=text_generator(), backend=self._opts.model
            )

            async for audio_chunk in audio_iterator:
                if audio_chunk:
                    output_emitter.push(audio_chunk)
                    self._mark_started()

        except WebSocketErr as e:
            logger.error(
                "Fish Audio WebSocket error during streaming",
                exc_info=e,
                extra={
                    "latency_mode": self._opts.latency_mode,
                    "format": self._opts.output_format,
                    "reference_id": self._opts.reference_id,
                },
            )
            raise APIConnectionError(f"Fish Audio WebSocket error: {e}") from e
