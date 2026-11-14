"""
*   Telnyx STT API documentation:
    <https://developers.telnyx.com/docs/voice/programmable-voice/stt-standalone>.
"""

from __future__ import annotations

import struct
import weakref
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, assert_never
from urllib.parse import urlencode

from livekit import rtc
from livekit.agents import APIConnectOptions, LanguageCode, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .common import NUM_CHANNELS, SAMPLE_RATE, STT_ENDPOINT, get_api_key

TranscriptionEngine = Literal["telnyx", "google", "deepgram", "azure"]


@dataclass
class _STTOptions:
    api_key: str
    language: LanguageCode
    transcription_engine: TranscriptionEngine
    interim_results: bool
    base_url: str
    sample_rate: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        transcription_engine: TranscriptionEngine = "telnyx",
        interim_results: bool = True,
        api_key: str | None = None,
        base_url: str = STT_ENDPOINT,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
            )
        )

        self._opts = _STTOptions(
            api_key=get_api_key(api_key),
            language=LanguageCode(language),
            transcription_engine=transcription_engine,
            interim_results=interim_results,
            base_url=base_url,
            sample_rate=sample_rate,
        )
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.transcription_engine

    @property
    def provider(self) -> str:
        return "telnyx"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        resolved_language = LanguageCode(language) if is_given(language) else self._opts.language

        stream = self.stream(language=language, conn_options=conn_options)
        try:
            frames = buffer if isinstance(buffer, list) else [buffer]
            for frame in frames:
                stream.push_frame(frame)
            stream.end_input()

            final_text = ""
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if event.alternatives:
                        final_text += event.alternatives[0].text

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=resolved_language,
                        text=final_text,
                    )
                ],
            )
        finally:
            await stream.aclose()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        resolved_language = LanguageCode(language) if is_given(language) else self._opts.language
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=resolved_language,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


def _create_streaming_wav_header(sample_rate: int, num_channels: int) -> bytes:
    """Create a WAV header for streaming with maximum possible size."""
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = 0x7FFFFFFF
    file_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header


class _Transcript(TypedDict):
    transcript: NotRequired[str]
    is_final: NotRequired[bool]
    confidence: NotRequired[float]


class SpeechStream(stt.WebSocketStream[_Transcript, None]):
    frame_ms = 50

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        language: LanguageCode,
        opts: _STTOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        opts = self._opts
        params = {
            "transcription_engine": opts.transcription_engine,
            "language": self._language,
            "input_format": "wav",
        }
        base_url = opts.base_url
        if base_url.startswith("http"):
            base_url = base_url.replace("http", "ws", 1)
        return stt.WebSocketTarget(
            f"{base_url}?{urlencode(params)}",
            headers={"Authorization": f"Bearer {opts.api_key}"},
        )

    def setup(self) -> bytes:
        return _create_streaming_wav_header(self._opts.sample_rate, NUM_CHANNELS)

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | None:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush() | stt.EndOfAudio():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Transcript) -> None:
        if text := message.get("transcript", ""):
            self.report(
                stt.Transcript(
                    text,
                    self._language,
                    final=message.get("is_final", False),
                    confidence=message.get("confidence", 0.0),
                )
            )
