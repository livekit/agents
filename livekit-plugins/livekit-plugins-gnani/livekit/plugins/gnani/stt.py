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

"""Speech-to-Text implementation for Gnani Vachana

This module provides an STT implementation that uses the Gnani Vachana API,
supporting both REST recognition and real-time streaming (WebSocket).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.misc import is_given

from .log import logger

GnaniSTTFormat = Literal["verbatim", "transcribe"]

GNANI_STT_BASE_URL = "https://api.vachana.ai"

GnaniSTTLanguages = Literal[
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "en-IN,hi-IN",
]

SUPPORTED_LANGUAGES: set[str] = {
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "en-IN,hi-IN",
}

STREAM_SUPPORTED_LANGUAGES: set[str] = {
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
}

SAMPLE_RATE_16K = 16000
SAMPLE_RATE_8K = 8000
STREAM_CHUNK_BYTES = 1024


@dataclass
class GnaniSTTOptions:
    api_key: str
    language: str
    sample_rate: int = SAMPLE_RATE_16K
    base_url: str = GNANI_STT_BASE_URL
    preferred_language: str | None = None
    format: str = "verbatim"
    itn_native_numerals: bool = False


_DEPRECATED_STT_KWARGS = frozenset(("organization_id", "user_id", "http_session"))


def _check_deprecated_args(kwargs: dict[str, Any]) -> None:
    """Warn about deprecated kwargs and raise on truly unknown ones."""
    for name in _DEPRECATED_STT_KWARGS:
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and no longer used")

    unknown = set(kwargs) - _DEPRECATED_STT_KWARGS
    if unknown:
        raise TypeError(
            f"STT.__init__() got unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        )


class STT(stt.STT):
    """Gnani Vachana Speech-to-Text implementation.

    Provides speech-to-text functionality using Gnani's Vachana platform.
    Supports REST recognition and real-time streaming via WebSocket.

    Args:
        language: BCP-47 language code (e.g. "hi-IN", "en-IN").
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        sample_rate: Audio sample rate for streaming (8000 or 16000).
        base_url: Vachana API base URL.
        preferred_language: Force single-language model for this code.
        format: "verbatim" (default) or "transcribe" (enables ITN).
        itn_native_numerals: Render digits in native script when format="transcribe".
    """

    def __init__(
        self,
        *,
        language: str = "en-IN",
        api_key: str | None = None,
        sample_rate: int = SAMPLE_RATE_16K,
        base_url: str = GNANI_STT_BASE_URL,
        preferred_language: str | None = None,
        format: GnaniSTTFormat = "verbatim",
        itn_native_numerals: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=False,
                aligned_transcript=False,
            )
        )

        _check_deprecated_args(kwargs)

        self._api_key = api_key or os.environ.get("GNANI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gnani API key is required. "
                "Provide it directly or set GNANI_API_KEY environment variable."
            )

        if sample_rate not in (SAMPLE_RATE_8K, SAMPLE_RATE_16K):
            raise ValueError("sample_rate must be 8000 or 16000")

        self._opts = GnaniSTTOptions(
            api_key=self._api_key,
            language=language,
            sample_rate=sample_rate,
            base_url=base_url,
            preferred_language=preferred_language,
            format=format,
            itn_native_numerals=itn_native_numerals,
        )
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "vachana-stt-v3"

    @property
    def provider(self) -> str:
        return "Gnani"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    @staticmethod
    def _single_attempt(conn_options: APIConnectOptions) -> APIConnectOptions:
        return APIConnectOptions(
            max_retry=0,
            retry_interval=conn_options.retry_interval,
            timeout=conn_options.timeout,
        )

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        return await super().recognize(
            buffer,
            language=language,
            conn_options=self._single_attempt(conn_options),
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form_data = aiohttp.FormData()
        form_data.add_field("audio_file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form_data.add_field("language_code", lang)
        form_data.add_field("format", self._opts.format)

        if self._opts.preferred_language is not None:
            form_data.add_field("preferred_language", self._opts.preferred_language)
        if self._opts.itn_native_numerals:
            form_data.add_field("itn_native_numerals", "true")

        headers: dict[str, str] = {
            "X-API-Key-ID": self._opts.api_key,
        }

        try:
            async with self._ensure_session().post(
                url=f"{self._opts.base_url}/stt/v3",
                data=form_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error(f"Gnani STT API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Gnani STT API Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                response_json = await res.json()
                transcript = response_json.get("transcript", "")
                request_id = response_json.get("request_id", "")

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[
                        stt.SpeechData(
                            language=LanguageCode(lang),
                            text=transcript,
                            confidence=1.0,
                        )
                    ],
                )

        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani STT API request timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani STT error: {e}") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        return SpeechStream(
            stt=self,
            opts=opts,
            conn_options=self._single_attempt(conn_options),
        )

    async def aclose(self) -> None:
        pass


class SpeechStream(stt.RecognizeStream):
    """WebSocket-based streaming STT for Gnani Vachana.

    Connects to wss://api.vachana.ai/stt/v3/stream and sends raw PCM audio
    in 1024-byte chunks (512 samples, 16-bit mono).
    """

    def __init__(
        self,
        *,
        stt: STT,
        opts: GnaniSTTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=opts.sample_rate,
        )
        self._opts = opts

    def _build_ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = "wss://" + base
        return f"{ws_base}/stt/v3/stream"

    async def _run(self) -> None:
        import websockets

        ws_url = self._build_ws_url()
        headers: dict[str, str] = {
            "x-api-key-id": self._opts.api_key,
            "lang_code": self._opts.language,
            "x-sample-rate": str(self._opts.sample_rate),
        }
        if self._opts.format != "verbatim":
            headers["x-format"] = self._opts.format
        if self._opts.preferred_language is not None:
            headers["preferred_language"] = self._opts.preferred_language
        if self._opts.itn_native_numerals:
            headers["itn_native_numerals"] = "true"

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                connected_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                connected_data = json.loads(connected_msg)
                if connected_data.get("type") != "connected":
                    logger.warning(f"Unexpected first message from Gnani STT: {connected_data}")

                send_task = asyncio.create_task(self._send_audio(ws), name="gnani-stt-send")
                recv_task = asyncio.create_task(self._recv_messages(ws), name="gnani-stt-recv")

                try:
                    # Wait for send to finish; if recv errors first, propagate it.
                    done, _ = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task.result()

                    if send_task.done() and not recv_task.done():
                        # All audio sent. The Gnani API has no application-level
                        # end-of-stream message, so give the server a short
                        # window to flush final transcripts before closing.
                        try:
                            await asyncio.wait_for(asyncio.shield(recv_task), timeout=1.0)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    await utils.aio.gracefully_cancel(send_task, recv_task)

        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"Gnani STT WebSocket closed unexpectedly: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani STT WebSocket connection timed out") from e
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani STT WebSocket error: {e}") from e

    async def _send_audio(self, ws: Any) -> None:
        audio_buffer = bytearray()

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                if audio_buffer:
                    await ws.send(bytes(audio_buffer))
                    audio_buffer.clear()
                continue

            frame: rtc.AudioFrame = data
            raw_pcm = frame.data.tobytes()
            audio_buffer.extend(raw_pcm)

            while len(audio_buffer) >= STREAM_CHUNK_BYTES:
                chunk = bytes(audio_buffer[:STREAM_CHUNK_BYTES])
                audio_buffer = audio_buffer[STREAM_CHUNK_BYTES:]
                await ws.send(chunk)

        if audio_buffer:
            await ws.send(bytes(audio_buffer))

    async def _recv_messages(self, ws: Any) -> None:
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue

                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "transcript":
                    text = data.get("text", "")
                    if not text:
                        continue

                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=data.get("segment_id", ""),
                            alternatives=[
                                stt.SpeechData(
                                    language=LanguageCode(self._opts.language),
                                    text=text,
                                    confidence=1.0,
                                )
                            ],
                        )
                    )

                elif msg_type in ("speech_start", "vad_start"):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                        )
                    )

                elif msg_type in ("speech_end", "vad_end"):
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.END_OF_SPEECH,
                        )
                    )

                elif msg_type == "processing":
                    pass

                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"Gnani STT stream error: {error_msg}")
                    raise APIStatusError(
                        message=f"Gnani STT stream error: {error_msg}",
                        status_code=500,
                        body=error_msg,
                    )

        except asyncio.CancelledError:
            raise
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Error receiving Gnani STT messages: {e}") from e
