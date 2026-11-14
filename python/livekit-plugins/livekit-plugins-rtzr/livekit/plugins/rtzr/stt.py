# Copyright 2023 LiveKit, Inc.
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

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TypedDict

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    http,
    stt,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.voice.io import TranscriptChunk

from .log import logger
from .rtzrapi import DEFAULT_SAMPLE_RATE, RTZROpenAPIClient

_DEFAULT_CHUNK_MS = 100


class _Word(TypedDict, total=False):
    text: str | None
    start_at: float | None
    duration: float | None


class _Alternative(TypedDict, total=False):
    text: str | None


class _Transcript(TypedDict, total=False):
    type: str
    error: object
    message: str
    start_at: float | None
    duration: float | None
    words: list[_Word] | None
    alternatives: list[_Alternative] | None
    final: bool | None


@dataclass
class _STTOptions:
    model_name: str = "sommers_ko"  # sommers_ko: "ko", sommers_ja: "ja"
    language: LanguageCode = LanguageCode("ko")  # ko, ja, en
    sample_rate: int = DEFAULT_SAMPLE_RATE
    encoding: str = "LINEAR16"  # or "OGG_OPUS" in future
    domain: str = "CALL"  # CALL, MEETING
    epd_time: float = 0.8  # endpoint detection time in seconds
    noise_threshold: float = 0.60
    active_threshold: float = 0.80
    use_punctuation: bool = False
    keywords: list[str] | list[tuple[str, float]] | None = None


class STT(stt.STT):
    """RTZR Streaming STT over WebSocket."""

    def __init__(
        self,
        *,
        model: str = "sommers_ko",
        language: str = "ko",
        sample_rate: int = 8000,
        domain: str = "CALL",
        epd_time: float = 0.8,
        noise_threshold: float = 0.60,
        active_threshold: float = 0.80,
        use_punctuation: bool = False,
        keywords: list[str] | list[tuple[str, float]] | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                # word timestamps don't seem to work despite the docs saying they do
                aligned_transcript="chunk",
                offline_recognize=False,
            )
        )

        self._params = _STTOptions(
            model_name=model,
            language=LanguageCode(language),
            sample_rate=sample_rate,
            domain=domain,
            epd_time=epd_time,
            noise_threshold=noise_threshold,
            active_threshold=active_threshold,
            use_punctuation=use_punctuation,
            keywords=keywords,
        )
        if keywords and model != "sommers_ko":
            logger.warning("RTZR keyword boosting is only supported with sommers_ko model")
        self._client = RTZROpenAPIClient(http_session=http_session)

    @property
    def model(self) -> str:
        return self._params.model_name

    @property
    def provider(self) -> str:
        return "RTZR"

    async def aclose(self) -> None:
        """Close the RTZR client and cleanup resources."""
        await self._client.close()

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Single-shot recognition is not supported; use stream().")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        return SpeechStream(
            stt=self,
            params=self._params,
            client=self._client,
            conn_options=conn_options,
        )


class SpeechStream(stt.SocketStream[_Transcript, None]):
    # ends a segment: the server sends what it holds, then closes
    goodbye = "EOS"
    frame_ms = _DEFAULT_CHUNK_MS

    def __init__(
        self,
        *,
        stt: STT,
        params: _STTOptions,
        client: RTZROpenAPIClient,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=params.sample_rate)
        self._params = params
        self._client = client
        self._segment_audio = False

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._segment_audio = False
        config = self._client.build_config(
            model_name=self._params.model_name,
            domain=self._params.domain,
            sample_rate=self._params.sample_rate,
            encoding=self._params.encoding,
            epd_time=self._params.epd_time,
            noise_threshold=self._params.noise_threshold,
            active_threshold=self._params.active_threshold,
            use_punctuation=self._params.use_punctuation,
            keywords=self._params.keywords,
        )
        return await self._client.streaming_target(config, conn_options=conn_options)

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[None]:
        if isinstance(item, rtc.AudioFrame):
            self._segment_audio = True
            return bytes(item.data)
        if (
            isinstance(item, stt.RecognizeStream.Flush)
            and self._segment_audio
            and not self.input.closed
        ):
            # each segment is one connection: EOS, then the next segment connects again (the
            # input's end sends its own EOS)
            self.reconnect("segment ended")
        return None

    def on_message(self, message: _Transcript) -> stt.Transcript | None:
        if "error" in message or message.get("type") == "error":
            raise APIStatusError(
                "RTZR STT server error",
                status_code=500,
                detail=str(message.get("error", message.get("message"))),
            )
        alternatives = message.get("alternatives")
        text = (alternatives[0].get("text") or "") if alternatives else ""
        if not text:
            return None
        offset = self.start_time_offset
        start = (message.get("start_at") or 0) / 1000.0 + offset
        words = message.get("words") or []
        return stt.Transcript(
            text,
            self._params.language,
            final=bool(message.get("final", False)),
            start_time=start,
            end_time=start + (message.get("duration") or 0) / 1000.0,
            words=[
                TranscriptChunk(
                    text=w.get("text") or "",
                    start_time=(w.get("start_at") or 0) / 1000.0 + offset,
                    end_time=((w.get("start_at") or 0) + (w.get("duration") or 0)) / 1000.0
                    + offset,
                )
                for w in words
            ]
            or None,
        )
