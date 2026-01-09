# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speech-to-Text implementation for Sarvam.ai

This module provides an STT implementation that uses the Sarvam.ai API.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import os
import platform
import weakref
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.misc import is_given

from .log import logger
from .version import __version__

# Sarvam API details
SARVAM_STT_BASE_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_STREAMING_URL = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_STT_TRANSLATE_BASE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_STT_TRANSLATE_STREAMING_URL = "wss://api.sarvam.ai/speech-to-text-translate/ws"

# Models
SarvamSTTModels = Literal["saarika:v2.5", "saarika:v2.0", "saaras:v2.5"]


def _build_custom_headers(
    api_key: str,
    additional_headers: dict | None = None,
) -> dict:
    """Build custom headers to identify the source of requests.

    Standardized headers sent across all flows (HTTP and WebSocket)
    for consistent debugging and request tracking.

    Args:
        api_key: Sarvam API key
        additional_headers: Additional headers to include (e.g., Content-Type)
    """
    headers = {
        "api-subscription-key": api_key,
        "User-Agent": f"Livekit/{__version__} Python/{platform.python_version()}",
    }

    if additional_headers:
        headers.update(additional_headers)
    return headers


class ConnectionState(enum.Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class SarvamSTTOptions:
    """Options for the Sarvam.ai STT service.

    Args:
        language: BCP-47 language code, e.g., "hi-IN", "en-IN"
        model: The Sarvam STT model to use
        base_url: API endpoint URL (auto-determined from model if not provided)
        streaming_url: WebSocket streaming URL (auto-determined from model if
            not provided)
        prompt: Optional prompt for STT translate (saaras models only)
    """

    language: str  # BCP-47 language code, e.g., "hi-IN", "en-IN"
    api_key: str
    model: SarvamSTTModels | str = "saarika:v2.5"
    base_url: str | None = None
    streaming_url: str | None = None
    prompt: str | None = None  # Optional prompt for STT translate (saaras models only)
    high_vad_sensitivity: bool | None = None
    sample_rate: int = 16000
    flush_signal: bool | None = None
    input_audio_codec: str | None = None

    def __post_init__(self) -> None:
        """Set URLs based on model if not explicitly provided."""
        if self.base_url is None or self.streaming_url is None:
            base_url, streaming_url = _get_urls_for_model(self.model)
            if self.base_url is None:
                self.base_url = base_url
            if self.streaming_url is None:
                self.streaming_url = streaming_url
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")


def _get_urls_for_model(model: str) -> tuple[str, str]:
    """Get base URL and streaming URL based on model type.

    Args:
        model: The Sarvam model name

    Returns:
        Tuple of (base_url, streaming_url)
    """
    if model.startswith("saaras:"):
        return (
            SARVAM_STT_TRANSLATE_BASE_URL,
            SARVAM_STT_TRANSLATE_STREAMING_URL,
        )
    else:  # saarika models
        return SARVAM_STT_BASE_URL, SARVAM_STT_STREAMING_URL


# TODO: Copied from livekit/agents/utils/audio.py, check if it can be reused
def _calculate_audio_duration(buffer: AudioBuffer) -> float:
    """Calculate audio duration from buffer."""
    try:
        if isinstance(buffer, list):
            # Calculate total duration from all frames
            total_samples = sum(frame.samples_per_channel for frame in buffer)
            if buffer and total_samples > 0:
                sample_rate = buffer[0].sample_rate
                return total_samples / sample_rate
        elif hasattr(buffer, "duration"):
            return buffer.duration / 1000.0  # buffer.duration is in ms
        elif hasattr(buffer, "samples_per_channel") and hasattr(buffer, "sample_rate"):
            # Single AudioFrame
            return buffer.samples_per_channel / buffer.sample_rate
    except Exception as e:
        logger.warning(f"Could not calculate audio duration: {e}")
    return 0.0


def _build_websocket_url(base_url: str, opts: SarvamSTTOptions) -> str:
    """Build WebSocket URL with parameters."""
    params = {
        "language-code": opts.language,
        "model": opts.model,
        "vad_signals": "true",
    }

    if opts.sample_rate:
        params["sample_rate"] = str(opts.sample_rate)
    if opts.high_vad_sensitivity is not None:
        params["high_vad_sensitivity"] = str(opts.high_vad_sensitivity).lower()
    if opts.flush_signal is not None:
        params["flush_signal"] = str(opts.flush_signal).lower()
    if opts.input_audio_codec:
        params["input_audio_codec"] = opts.input_audio_codec

    return f"{base_url}?{urlencode(params)}"


class STT(stt.STT):
    """Sarvam.ai Speech-to-Text implementation.

    This class provides speech-to-text functionality using the Sarvam.ai API.
    Sarvam.ai specializes in high-quality STT for Indian languages.

    Args:
        language: BCP-47 language code, e.g., "hi-IN", "en-IN"
        model: The Sarvam STT model to use
        api_key: Sarvam.ai API key (falls back to SARVAM_API_KEY env var)
        base_url: API endpoint URL
        http_session: Optional aiohttp session to use
        prompt: Optional prompt for STT translate (saaras models only)
    """

    def __init__(
        self,
        *,
        language: str = "en-IN",
        model: SarvamSTTModels | str = "saarika:v2.5",
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        prompt: str | None = None,
        high_vad_sensitivity: bool | None = None,
        sample_rate: int = 16000,
        flush_signal: bool | None = None,
        input_audio_codec: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                # chunk timestamps don't seem to work despite the docs saying they do
                aligned_transcript=False,
            )
        )

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. Provide it directly or set SARVAM_API_KEY env var."
            )

        self._opts = SarvamSTTOptions(
            language=language,
            api_key=self._api_key,
            model=model,
            base_url=base_url,
            prompt=prompt,
            high_vad_sensitivity=high_vad_sensitivity,
            sample_rate=sample_rate,
            flush_signal=flush_signal,
            input_audio_codec=input_audio_codec,
        )
        self._session = http_session
        self._logger = logger.getChild(self.__class__.__name__)
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sarvam"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[SarvamSTTModels | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech using Sarvam.ai API.

        Args:
            buffer: Audio buffer containing speech data
            language: BCP-47 language code (overrides the one set in
                constructor)
            model: Sarvam model to use (overrides the one set in constructor)
            conn_options: Connection options for API requests

        Returns:
            A SpeechEvent containing the transcription result

        Raises:
            APIConnectionError: On network connection errors
            APIStatusError: On API errors (non-200 status)
            APITimeoutError: On API timeout
        """
        opts_language = self._opts.language if isinstance(language, type(NOT_GIVEN)) else language
        opts_model = self._opts.model if isinstance(model, type(NOT_GIVEN)) else model

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            wav_bytes,
            filename="audio.wav",
            content_type="audio/wav",
        )

        # Add model and language_code to the form data if specified
        # Sarvam API docs state language_code is optional for saarika:v2x but
        # mandatory for v1
        # Model is also optional, defaults to saarika:v2.5
        if opts_language:
            form_data.add_field("language_code", opts_language)
        if opts_model:
            form_data.add_field("model", str(opts_model))

        if not self._api_key:
            raise ValueError("API key cannot be None")
        headers = _build_custom_headers(self._api_key)

        try:
            if self._opts.base_url is None:
                raise ValueError("base_url cannot be None")
            async with self._ensure_session().post(
                url=self._opts.base_url,
                data=form_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                raw_text = await res.text()
                body = _json_loads_maybe(raw_text)

                if res.status != 200:
                    req_id = _extract_request_id(body)
                    self._logger.error(
                        "Sarvam STT HTTP error",
                        extra={
                            "status_code": res.status,
                            "request_id": req_id,
                            "body": body,
                            "model": str(opts_model),
                            "language": str(opts_language),
                        },
                    )
                    raise APIStatusError(
                        message="Sarvam STT request failed",
                        status_code=res.status,
                        request_id=req_id,
                        body=body,
                    )

                if not isinstance(body, dict):
                    self._logger.error(
                        "Sarvam STT returned unexpected response format",
                        extra={
                            "status_code": res.status,
                            "body": body,
                            "model": str(opts_model),
                            "language": str(opts_language),
                        },
                    )
                    raise APIStatusError(
                        message="Sarvam STT unexpected response format",
                        status_code=res.status,
                        body=body,
                        retryable=True,
                    )

                self._logger.debug(
                    "Sarvam STT response received",
                    extra={
                        "status_code": res.status,
                        "request_id": body.get("request_id"),
                        "model": str(opts_model),
                        "language": str(opts_language),
                        "sdk_source": "livekit-agents-python",
                        "sdk_version": __version__,
                    },
                )

                transcript_text = body.get("transcript", "") or ""
                request_id = body.get("request_id", "") or ""
                detected_language = body.get("language_code")
                if not isinstance(detected_language, str):
                    detected_language = opts_language or ""

                start_time = 0.0
                end_time = 0.0

                # Try to get timestamps if available
                timestamps_data = body.get("timestamps")
                if isinstance(timestamps_data, dict):
                    words_ts_start = timestamps_data.get("start_time_seconds")
                    words_ts_end = timestamps_data.get("end_time_seconds")
                    if isinstance(words_ts_start, list) and words_ts_start:
                        start_time = float(words_ts_start[0] or 0.0)
                    if isinstance(words_ts_end, list) and words_ts_end:
                        end_time = float(words_ts_end[-1] or 0.0)

                # If start/end times are still 0, use buffer duration as an
                # estimate for end_time
                if start_time == 0.0 and end_time == 0.0:
                    end_time = _calculate_audio_duration(buffer)

                alternatives = [
                    stt.SpeechData(
                        language=detected_language,
                        text=transcript_text,
                        start_time=start_time,
                        end_time=end_time,
                        # Sarvam doesn't provide confidence score in this
                        # response.
                        confidence=1.0,
                    )
                ]

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )

        except asyncio.CancelledError:
            raise
        except APIError:
            raise
        except asyncio.TimeoutError as e:
            self._logger.error(
                "Sarvam STT request timed out",
                extra={
                    "timeout": conn_options.timeout,
                    "model": str(opts_model),
                    "language": str(opts_language),
                },
                exc_info=True,
            )
            raise APITimeoutError("Sarvam STT request timed out") from e
        except aiohttp.ClientError as e:
            self._logger.error(
                "Sarvam STT connection error",
                extra={
                    "error_type": type(e).__name__,
                    "model": str(opts_model),
                    "language": str(opts_language),
                },
                exc_info=True,
            )
            raise APIConnectionError(f"Sarvam STT connection error: {e}") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[SarvamSTTModels | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        high_vad_sensitivity: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        flush_signal: NotGivenOr[bool] = NOT_GIVEN,
        input_audio_codec: NotGivenOr[str] = NOT_GIVEN,
    ) -> SpeechStream:
        """Create a streaming transcription session."""
        opts_language = language if is_given(language) else self._opts.language
        opts_model = model if is_given(model) else self._opts.model

        if not isinstance(opts_language, str):
            opts_language = self._opts.language
        if not isinstance(opts_model, str):
            opts_model = self._opts.model

        # Handle prompt conversion from NotGiven to None
        final_prompt: str | None
        if isinstance(prompt, str):
            final_prompt = prompt
        else:
            final_prompt = self._opts.prompt

        opts_high_vad = (
            high_vad_sensitivity
            if is_given(high_vad_sensitivity)
            else self._opts.high_vad_sensitivity
        )
        opts_sample_rate = sample_rate if is_given(sample_rate) else self._opts.sample_rate
        opts_flush_signal = flush_signal if is_given(flush_signal) else self._opts.flush_signal
        opts_sample_rate = sample_rate if is_given(sample_rate) else self._opts.sample_rate
        opts_flush_signal = flush_signal if is_given(flush_signal) else self._opts.flush_signal
        opts_input_codec = (
            input_audio_codec if is_given(input_audio_codec) else self._opts.input_audio_codec)

        # Create options for the stream
        stream_opts = SarvamSTTOptions(
            language=opts_language,
            api_key=self._api_key if self._api_key else "",
            model=opts_model,
            prompt=final_prompt,
            high_vad_sensitivity=opts_high_vad,
            sample_rate=opts_sample_rate,
            flush_signal=opts_flush_signal,
            input_audio_codec=opts_input_codec,
        )

        # Create a fresh session for this stream to avoid conflicts
        stream_session = aiohttp.ClientSession()

        if not self._api_key:
            raise ValueError("API key cannot be None")
        stream = SpeechStream(
            stt=self,
            opts=stream_opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=stream_session,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    """Sarvam.ai streaming speech-to-text implementation."""

    _CHUNK_DURATION_MS = 50

    def __init__(
        self,
        *,
        stt: STT,
        opts: SarvamSTTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self._opts = opts
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._logger = logger.getChild(self.__class__.__name__)
        self._reconnect_event = asyncio.Event()

        # Connection state management
        self._connection_state = ConnectionState.DISCONNECTED
        self._connection_lock = asyncio.Lock()
        self._session_id = id(self)

        # Request ID tracking
        self._client_request_id: str | None = None
        self._server_request_id: str | None = None

        # Add flush mechanism
        self._ws: aiohttp.ClientWebSocketResponse | None = (
            None  # Store WebSocket reference for flush
        )
        self._should_flush = False  # Flag to trigger flush

        # Task management for cleanup
        self._audio_task: asyncio.Task | None = None
        self._message_task: asyncio.Task | None = None
        self._audio_encoding = self._opts.input_audio_codec or "audio/wav"
        self._chunk_size = max(
            int(self._opts.sample_rate * self._CHUNK_DURATION_MS / 1000),
            1,
        )
        self._end_of_stream_msg = self._build_end_of_stream_message()

    def _build_end_of_stream_message(self) -> str:
        return json.dumps(
            {
                "type": "end_of_stream",
                "audio": {
                    "data": "",
                    "encoding": self._audio_encoding,
                    "sample_rate": self._opts.sample_rate,
                },
            }
        )

    async def aclose(self) -> None:
        """Close the stream and clean up resources."""
        self._logger.debug(
            "Starting stream cleanup",
            extra={"session_id": self._session_id},
        )

        async with self._connection_lock:
            self._connection_state = ConnectionState.DISCONNECTED

        # Cancel running tasks first
        tasks_to_cancel = []
        if self._audio_task and not self._audio_task.done():
            tasks_to_cancel.append(self._audio_task)
        if self._message_task and not self._message_task.done():
            tasks_to_cancel.append(self._message_task)

        if tasks_to_cancel:
            try:
                await utils.aio.cancel_and_wait(*tasks_to_cancel)
            except Exception as e:
                self._logger.warning(
                    f"Error cancelling tasks: {e}",
                    extra={"session_id": self._session_id},
                )

        # Close WebSocket
        try:
            if self._ws and not self._ws.closed:
                await self._ws.close()
                self._logger.debug(
                    "WebSocket closed",
                    extra={"session_id": self._session_id},
                )
        except Exception as e:
            self._logger.warning(
                f"Error closing WebSocket: {e}",
                extra={"session_id": self._session_id},
            )
        finally:
            self._ws = None

        # Call parent cleanup
        try:
            await super().aclose()
        except Exception as e:
            self._logger.warning(
                f"Error in parent cleanup: {e}",
                extra={"session_id": self._session_id},
            )

        # Close session last
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                self._logger.debug(
                    "HTTP session closed",
                    extra={"session_id": self._session_id},
                )
        except Exception as e:
            self._logger.warning(
                f"Error closing session: {e}",
                extra={"session_id": self._session_id},
            )
        finally:
            # Clear reference to help with garbage collection
            pass  # Session reference will be cleared when object is destroyed

    def update_options(
        self,
        *,
        language: str,
        model: str,
        prompt: str | None = None,
    ) -> None:
        """Update streaming options."""
        if not language or not language.strip():
            raise ValueError("Language cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model cannot be empty")

        self._opts.language = language
        self._opts.model = model
        if prompt is not None:
            self._opts.prompt = prompt
        self._logger.info(
            "Options updated, triggering reconnection",
            extra={
                "session_id": self._session_id,
                "language": language,
                "model": model,
                "prompt": prompt,
            },
        )
        self._reconnect_event.set()

    async def _send_initial_config(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send initial configuration message with prompt for saaras models."""
        try:
            config_message = {"prompt": self._opts.prompt, "type": "config"}
            await ws.send_str(json.dumps(config_message))
            self._logger.debug(
                "Sent initial config for saaras model",
                extra={
                    "session_id": self._session_id,
                    "prompt": self._opts.prompt,
                },
            )
        except Exception as e:
            self._logger.error(
                f"Failed to send initial configuration: {e}",
                extra={"session_id": self._session_id},
                exc_info=True,
            )
            raise APIConnectionError(f"Failed to send initial config: {e}") from e

    def _build_log_context(self) -> dict:
        """Build consistent logging context with request IDs."""
        return {
            "session_id": self._session_id,
            "client_request_id": self._client_request_id,
            "server_request_id": self._server_request_id,
            "model": self._opts.model,
            "language": self._opts.language,
            "sdk_source": "livekit-agents-python",
            "sdk_version": __version__,
        }

    async def _run(self) -> None:
        """
        Main loop for streaming transcription.

        Important: retries/backoff are handled by
        livekit.agents.stt.RecognizeStream._main_task()
        based on raised APIError subclasses, so this implementation should
        raise the right
        exceptions and avoid its own retry loops.
        """
        # Generate a client-side request ID for this stream
        self._client_request_id = utils.shortuuid()
        self._server_request_id = None

        self._logger.debug(
            "STT stream initialized",
            extra=self._build_log_context(),
        )

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                self._ws = ws
                self._closing_ws = False

                # Send initial configuration message for saaras models (STT
                # translate)
                if self._opts.model.startswith("saaras") and self._opts.prompt:
                    await self._send_initial_config(ws)

                # Create tasks for audio processing and message handling
                self._audio_task = asyncio.create_task(self._process_audio(ws))
                self._message_task = asyncio.create_task(self._process_messages(ws))

                tasks_group = asyncio.gather(
                    self._audio_task,
                    self._message_task,
                )
                reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if tasks_group in done:
                        tasks_group.result()
                        return

                    # reconnect requested
                    self._logger.info(
                        "Reconnection requested, closing current connection",
                        extra={"session_id": self._session_id},
                    )
                    self._reconnect_event.clear()
                    continue

                finally:
                    await utils.aio.gracefully_cancel(
                        self._audio_task,
                        self._message_task,
                        reconnect_task,
                    )
                    tasks_group.cancel()
                    with contextlib.suppress(Exception):
                        tasks_group.exception()

            finally:
                if ws is not None and not ws.closed:
                    await ws.close()
                ws = None
                self._ws = None

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to Sarvam STT WebSocket with correct error handling."""
        if self._session.closed:
            raise APIConnectionError("Session is closed, cannot establish WebSocket connection")

        async with self._connection_lock:
            self._connection_state = ConnectionState.CONNECTING

        if self._opts.streaming_url is None:
            raise ValueError("streaming_url cannot be None")
        ws_url = _build_websocket_url(self._opts.streaming_url, self._opts)

        # Connect to WebSocket with proper authentication
        headers = {"api-subscription-key": self._api_key}

        self._logger.info(
            "Connecting to Sarvam STT WebSocket",
            extra={
                **self._build_log_context(),
                "url": ws_url,
            },
        )

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(ws_url, headers=headers),
                self._conn_options.timeout,
            )
        except asyncio.CancelledError:
            raise
        except aiohttp.WSServerHandshakeError as e:
            # Handshake status is meaningful: 4xx should not be retried by
            # default.
            status = getattr(e, "status", -1) or -1
            self._logger.error(
                "Sarvam STT WebSocket handshake failed",
                extra={
                    "session_id": self._session_id,
                    "status_code": status,
                    "error_type": type(e).__name__,
                    "url": ws_url,
                },
                exc_info=True,
            )
            raise APIStatusError(
                message="Sarvam STT WebSocket handshake failed",
                status_code=status,
                body=str(e),
            ) from e
        except asyncio.TimeoutError as e:
            self._logger.error(
                "Timed out connecting to Sarvam STT WebSocket",
                extra={
                    "session_id": self._session_id,
                    "timeout": self._conn_options.timeout,
                    "url": ws_url,
                },
                exc_info=True,
            )
            raise APITimeoutError("Timed out connecting to Sarvam STT WebSocket") from e
        except aiohttp.ClientError as e:
            self._logger.error(
                "Failed to connect to Sarvam STT WebSocket",
                extra={
                    "session_id": self._session_id,
                    "error_type": type(e).__name__,
                    "url": ws_url,
                },
                exc_info=True,
            )
            raise APIConnectionError(f"Failed to connect to Sarvam STT WebSocket: {e}") from e

        async with self._connection_lock:
            self._connection_state = ConnectionState.CONNECTED

        self._logger.info(
            "Sarvam STT WebSocket connected",
            extra={
                **self._build_log_context(),
                "ws_closed": ws.closed,
            },
        )
        return ws

    @utils.log_exceptions(logger=logger)
    async def _process_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Process audio frames and send them in chunks."""
        import base64

        import numpy as np

        # Audio buffering for chunked sending
        audio_buffer: list[np.int16] = []
        chunk_size = self._chunk_size  # Derived from selected sample rate
        chunks_sent = 0

        self._logger.debug(
            "Starting audio processing",
            extra={
                **self._build_log_context(),
                "chunk_size": chunk_size,
            },
        )

        try:
            async for frame in self._input_ch:
                if ws.closed:
                    raise APIConnectionError(
                        f"Sarvam STT WebSocket closed while sending audio (code={ws.close_code})"
                    )
                if isinstance(frame, rtc.AudioFrame):
                    try:
                        # Convert audio frame to Int16 data
                        audio_data = frame.data.tobytes()
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        audio_buffer.extend(audio_array)

                        # Check if we have enough data for a chunk
                        while len(audio_buffer) >= chunk_size:
                            # Convert to Int16Array
                            chunk_data = np.array(
                                audio_buffer[:chunk_size],
                                dtype=np.int16,
                            )

                            # Convert to base64
                            base64_audio = base64.b64encode(chunk_data.tobytes()).decode("utf-8")

                            # Send audio in the required format
                            audio_message = {
                                "audio": {
                                    "data": base64_audio,
                                    "encoding": self._audio_encoding,
                                    "sample_rate": self._opts.sample_rate,
                                }
                            }

                            try:
                                await ws.send_str(json.dumps(audio_message))
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                self._logger.error(
                                    "Failed to send audio chunk",
                                    extra={
                                        "session_id": self._session_id,
                                        "chunks_sent": chunks_sent,
                                        "ws_closed": ws.closed,
                                        "close_code": ws.close_code,
                                        "error_type": type(e).__name__,
                                    },
                                    exc_info=True,
                                )
                                raise APIConnectionError(
                                    f"Failed to send audio over WebSocket: {e}"
                                ) from e
                            chunks_sent += 1

                            # Remove sent data from buffer
                            audio_buffer = audio_buffer[chunk_size:]

                            # Log progress periodically
                            if chunks_sent % 100 == 0:
                                self._logger.debug(
                                    f"Sent {chunks_sent} audio chunks",
                                    extra={"session_id": self._session_id},
                                )

                    except Exception as e:
                        self._logger.error(
                            f"Error processing audio frame: {e}",
                            extra={"session_id": self._session_id},
                            exc_info=True,
                        )
                        raise

                elif isinstance(frame, self._FlushSentinel):
                    # LiveKit VAD FlushSentinel - handles stream termination
                    self._logger.debug(
                        "Received FlushSentinel, sending end of stream",
                        extra={"session_id": self._session_id},
                    )
                    self._closing_ws = True
                    try:
                        await ws.send_str(self._end_of_stream_msg)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._logger.error(
                            "Failed to send end_of_stream message",
                            extra={
                                "session_id": self._session_id,
                                "ws_closed": ws.closed,
                                "close_code": ws.close_code,
                                "error_type": type(e).__name__,
                            },
                            exc_info=True,
                        )
                        raise APIConnectionError(f"Failed to send end_of_stream: {e}") from e
                    break

                # Check if Sarvam VAD triggered flush
                if self._should_flush:
                    self._logger.debug(
                        "VAD triggered flush, sending flush message",
                        extra={"session_id": self._session_id},
                    )
                    flush_message = {"type": "flush"}
                    try:
                        await ws.send_str(json.dumps(flush_message))
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._logger.error(
                            "Failed to send flush message",
                            extra={
                                "session_id": self._session_id,
                                "ws_closed": ws.closed,
                                "close_code": ws.close_code,
                                "error_type": type(e).__name__,
                            },
                            exc_info=True,
                        )
                        raise APIConnectionError(f"Failed to send flush: {e}") from e
                    self._should_flush = False  # Reset flag

        except asyncio.CancelledError:
            raise
        except APIError:
            raise
        except Exception as e:
            self._logger.error(
                "Error in audio processing loop",
                extra={
                    "session_id": self._session_id,
                    "chunks_sent": chunks_sent,
                    "ws_closed": ws.closed,
                    "close_code": ws.close_code,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise APIConnectionError(f"Error in Sarvam STT audio loop: {e}") from e
        finally:
            self._logger.debug(
                f"Audio processing completed, sent {chunks_sent} chunks",
                extra={
                    **self._build_log_context(),
                    "chunks_sent": chunks_sent,
                },
            )

    @utils.log_exceptions(logger=logger)
    async def _process_messages(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Process incoming messages from the WebSocket."""
        self._logger.info(
            "Starting message processing",
            extra={
                **self._build_log_context(),
                "ws_closed": ws.closed,
            },
        )

        try:
            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if self._closing_ws or self._session.closed:
                        self._logger.info(
                            "WebSocket closed (expected)",
                            extra={
                                "session_id": self._session_id,
                                "close_code": ws.close_code,
                                "ws_closed": ws.closed,
                            },
                        )
                        return

                    self._logger.error(
                        "WebSocket closed unexpectedly",
                        extra={
                            "session_id": self._session_id,
                            "close_code": ws.close_code,
                            "ws_closed": ws.closed,
                        },
                    )
                    raise APIConnectionError(
                        f"Sarvam STT WebSocket closed unexpectedly (code={ws.close_code})"
                    )

                if msg.type == aiohttp.WSMsgType.ERROR:
                    err = ws.exception()
                    self._logger.error(
                        "WebSocket error",
                        extra={
                            "session_id": self._session_id,
                            "close_code": ws.close_code,
                            "ws_closed": ws.closed,
                            "ws_exception": str(err),
                        },
                        exc_info=True,
                    )
                    raise APIConnectionError(f"Sarvam STT WebSocket error: {err}")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    self._logger.warning(
                        "Unexpected WebSocket message type",
                        extra={
                            "session_id": self._session_id,
                            "msg_type": str(msg.type),
                        },
                    )
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    self._logger.warning(
                        "Invalid JSON received from WebSocket",
                        extra={"session_id": self._session_id, "data": msg.data},
                    )
                    continue

                await self._handle_message(data)

        except asyncio.CancelledError:
            raise
        except APIError:
            raise
        except Exception as e:
            self._logger.error(
                "Error in message processing loop",
                extra={
                    "session_id": self._session_id,
                    "ws_closed": ws.closed,
                    "close_code": ws.close_code,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise APIConnectionError(f"Error in Sarvam STT message loop: {e}") from e

    async def _handle_message(self, data: dict) -> None:
        """Handle different types of messages from Sarvam streaming API."""
        try:
            msg_type = data.get("type")
            if not msg_type:
                self._logger.warning(
                    "Received message without type field",
                    extra={"session_id": self._session_id, "data": data},
                )
                return

            if msg_type == "data":
                await self._handle_transcript_data(data)
            elif msg_type == "events":
                await self._handle_events(data)
            elif msg_type == "error":
                await self._handle_error_message(data)
            else:
                self._logger.debug(
                    f"Unknown message type: {msg_type}",
                    extra={"session_id": self._session_id, "data": data},
                )

        except KeyError as e:
            self._logger.warning(
                f"Missing required field in message: {e}",
                extra={"session_id": self._session_id, "data": data},
            )
        except Exception as e:
            self._logger.error(
                f"Unexpected error handling message: {e}",
                extra={"session_id": self._session_id, "data": data},
                exc_info=True,
            )
            raise APIStatusError(f"Message processing error: {e}") from e

    async def _handle_transcript_data(self, data: dict) -> None:
        """Handle transcription result messages."""
        transcript_data = data.get("data", {})
        transcript_text = transcript_data.get("transcript", "")
        language = transcript_data.get("language_code", "")
        request_id = transcript_data.get("request_id", "")

        # Capture server request ID if not already set
        if request_id and not self._server_request_id:
            self._server_request_id = request_id
            self._logger.debug(
                "Received Sarvam server request ID",
                extra=self._build_log_context(),
            )

        if not transcript_text:
            self._logger.debug(
                "Received empty transcript",
                extra=self._build_log_context(),
            )
            return

        try:
            # Create usage event with proper metrics extraction
            metrics = transcript_data.get("metrics", {})
            request_data = {
                "client_request_id": self._client_request_id,
                "server_request_id": request_id,
                "processing_latency": metrics.get("processing_latency", 0.0),
            }
            usage_event = stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=json.dumps(request_data),
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=metrics.get("audio_duration", 0.0),
                ),
            )
            self._event_ch.send_nowait(usage_event)

            # Create speech data
            speech_data = stt.SpeechData(
                language=language,
                text=transcript_text,
                start_time=transcript_data.get("speech_start", 0.0),
                end_time=transcript_data.get("speech_end", 0.0),
            )

            # Create final transcript event with request_id
            # Use server request_id as primary, but log both IDs for tracking
            speech_event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,  # Sarvam server request ID
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(speech_event)

            self._logger.debug(
                "Transcript processed successfully",
                extra={
                    **self._build_log_context(),
                    "text_length": len(transcript_text),
                    "language": language,
                    "confidence": speech_data.confidence,
                },
            )

        except Exception as e:
            self._logger.error(
                f"Error processing transcript data: {e}",
                extra={
                    "session_id": self._session_id,
                    "transcript_data": transcript_data,
                },
                exc_info=True,
            )
            raise

    async def _handle_events(self, data: dict) -> None:
        """Handle VAD (Voice Activity Detection) events."""
        event_data = data.get("data", {})
        signal_type = event_data.get("signal_type")

        if not signal_type:
            self._logger.warning(
                "VAD event missing signal_type",
                extra={
                    **self._build_log_context(),
                    "event_data": event_data,
                },
            )
            return

        self._logger.debug(
            f"Processing VAD event: {signal_type}",
            extra={
                **self._build_log_context(),
                "signal_type": signal_type,
            },
        )

        try:
            if signal_type == "START_SPEECH":
                if not self._speaking:
                    self._speaking = True
                    start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    self._event_ch.send_nowait(start_event)
                    self._logger.debug(
                        "Speech started",
                        extra=self._build_log_context(),
                    )

            elif signal_type == "END_SPEECH":
                if self._speaking:
                    self._speaking = False
                    end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    self._event_ch.send_nowait(end_event)

                    # Set flag to trigger flush when Sarvam detects end of
                    # speech.
                    self._should_flush = True
                    self._logger.debug(
                        "Speech ended, flush triggered",
                        extra=self._build_log_context(),
                    )
            else:
                self._logger.debug(
                    f"Unknown VAD signal type: {signal_type}",
                    extra=self._build_log_context(),
                )

        except Exception as e:
            self._logger.error(
                f"Error processing VAD event: {e}",
                extra={
                    "session_id": self._session_id,
                    "event_data": event_data,
                },
                exc_info=True,
            )
            raise

    async def _handle_error_message(self, data: dict) -> None:
        """Handle error messages from the API."""
        error_info = data.get("error", "Unknown error")
        error_code = data.get("code", "unknown")

        self._logger.error(
            f"API error received: {error_info}",
            extra={
                "session_id": self._session_id,
                "error_code": error_code,
                "error_info": error_info,
            },
        )

        # Determine if error is recoverable based on error code/type
        recoverable_codes = ["rate_limit", "temporary_unavailable", "timeout"]
        recoverable_keywords = [
            "rate limit",
            "temporary",
            "timeout",
            "connection",
        ]

        is_recoverable = error_code in recoverable_codes or any(
            keyword in str(error_info).lower() for keyword in recoverable_keywords
        )

        if is_recoverable:
            # Treat as retryable: transient, rate limit, temporary errors
            raise APIConnectionError(
                f"Sarvam STT recoverable streaming error (code={error_code}): {error_info}"
            )
        else:
            # Treat as non-retryable by default: invalid args, auth, etc.
            raise APIStatusError(
                message=(f"Sarvam STT streaming error (code={error_code}): {error_info}"),
                status_code=-1,
                body=data,
                retryable=False,
            )
