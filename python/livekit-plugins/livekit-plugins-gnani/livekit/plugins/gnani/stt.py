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

import os
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, NotRequired, TypedDict, assert_never, cast

from pydantic import Field

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    http,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import GnaniSTTLanguages

GnaniSTTFormat = Literal["verbatim", "transcribe"]

GNANI_STT_BASE_URL = "https://api.vachana.ai"

SAMPLE_RATE_16K = 16000
SAMPLE_RATE_8K = 8000
SAMPLE_RATE_44K = 44100
SAMPLE_RATE_48K = 48000
STREAM_SUPPORTED_SAMPLE_RATES = (
    SAMPLE_RATE_8K,
    SAMPLE_RATE_16K,
    SAMPLE_RATE_44K,
    SAMPLE_RATE_48K,
)
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
    use_streaming: bool = True


class STT(stt.STT):
    """Gnani Vachana Speech-to-Text implementation.

    Provides speech-to-text functionality using Gnani's Vachana platform.
    Supports REST recognition and real-time streaming via WebSocket.

    Args:
        language: BCP-47 language code (e.g. "hi-IN", "en-IN").
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        sample_rate: Audio sample rate for streaming (8000, 16000, 44100, or 48000).
        base_url: Vachana API base URL.
        preferred_language: Force single-language model for this code.
        format: "verbatim" (default) or "transcribe" (enables ITN).
        itn_native_numerals: Render digits in native script when format="transcribe".
        use_streaming: When True (default), transcribe over the WebSocket stream
            (wss://.../stt/v3/stream). When False, use REST recognition
            (POST /stt/v3), which requires a local VAD — LiveKit wraps the STT
            with ``stt.StreamAdapter`` automatically.
    """

    def __init__(
        self,
        *,
        language: GnaniSTTLanguages | str = "en-IN",
        api_key: str | None = None,
        sample_rate: int = SAMPLE_RATE_16K,
        base_url: str = GNANI_STT_BASE_URL,
        preferred_language: str | None = None,
        format: GnaniSTTFormat = "verbatim",
        itn_native_numerals: bool = False,
        use_streaming: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_streaming,
                interim_results=False,
                aligned_transcript=False,
            )
        )

        self._api_key = api_key or os.environ.get("GNANI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gnani API key is required. "
                "Provide it directly or set GNANI_API_KEY environment variable."
            )

        if sample_rate not in STREAM_SUPPORTED_SAMPLE_RATES:
            allowed = ", ".join(str(r) for r in STREAM_SUPPORTED_SAMPLE_RATES)
            raise ValueError(f"sample_rate must be one of {allowed}, got {sample_rate}")

        self._opts = GnaniSTTOptions(
            api_key=self._api_key,
            language=language,
            sample_rate=sample_rate,
            base_url=base_url,
            preferred_language=preferred_language,
            format=format,
            itn_native_numerals=itn_native_numerals,
            use_streaming=use_streaming,
        )
        self._session: http.HttpSession | None = None

    @property
    def model(self) -> str:
        return "vachana-stt-v3"

    @property
    def provider(self) -> str:
        return "Gnani"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    @staticmethod
    def _single_attempt(conn_options: APIConnectOptions) -> APIConnectOptions:
        return replace(conn_options, max_retry=0)

    async def recognize(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
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
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form_data = http.FormData()
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

        async with self._ensure_session().post(
            f"{self._opts.base_url}/stt/v3",
            data=form_data,
            headers=headers,
            conn_options=conn_options,
        ) as res:
            await res.raise_for_status()

            response_json = cast(dict[str, Any], await res.json())
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

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if not self._opts.use_streaming:
            return cast(
                "SpeechStream",
                super().stream(language=language, conn_options=conn_options),
            )

        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        return SpeechStream(stt=self, opts=opts, conn_options=conn_options)

    async def aclose(self) -> None:
        pass


class _Connected(TypedDict):
    type: Literal["connected"]


class _Transcript(TypedDict):
    type: Literal["transcript"]
    text: NotRequired[str]
    segment_id: NotRequired[str]


class _SpeechStart(TypedDict):
    type: Literal["speech_start"]


class _VadStart(TypedDict):
    type: Literal["vad_start"]


class _SpeechEnd(TypedDict):
    type: Literal["speech_end"]


class _VadEnd(TypedDict):
    type: Literal["vad_end"]


class _Error(TypedDict):
    type: Literal["error"]
    message: NotRequired[str]


_Message = Annotated[
    _Connected | _Transcript | _SpeechStart | _VadStart | _SpeechEnd | _VadEnd | _Error,
    Field(discriminator="type"),
]


class SpeechStream(stt.WebSocketStream[_Message, None]):
    """WebSocket-based streaming STT for Gnani Vachana.

    Connects to wss://api.vachana.ai/stt/v3/stream and sends raw PCM audio
    in chunks of about 1024 bytes (512 samples, 16-bit mono).
    """

    # the session says "connected" before it takes audio
    awaits_ready = True
    # speech_start and speech_end are Gnani's own speech boundaries
    provider_vad = True
    ping_interval = 20.0

    def __init__(
        self,
        *,
        stt: STT,
        opts: GnaniSTTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self.frame_ms = STREAM_CHUNK_BYTES // 2 * 1000 // opts.sample_rate

    def _build_ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = "wss://" + base
        return f"{ws_base}/stt/v3/stream"

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
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
        return stt.WebSocketTarget(self._build_ws_url(), headers=headers)

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | None:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush() | stt.EndOfAudio():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if message["type"] == "connected":
            self.report(stt.Ready())
        elif message["type"] == "error":
            raise APIError(
                "Gnani STT stream error", kind="invalid_request", detail=message.get("message")
            )
        elif message["type"] == "speech_start" or message["type"] == "vad_start":
            self.report(stt.SpeechStarted())
        elif message["type"] == "speech_end" or message["type"] == "vad_end":
            self.report(stt.SpeechEnded())
        elif message["type"] == "transcript":
            if text := message.get("text", ""):
                self.report(
                    stt.Transcript(
                        text,
                        LanguageCode(self._opts.language),
                        final=True,
                        request_id=message.get("segment_id", ""),
                        confidence=1.0,
                    )
                )
        else:
            assert_never(message)
