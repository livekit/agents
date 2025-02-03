from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, List, Literal, TypedDict, Union

from livekit import rtc

from ..utils import AudioBuffer, aio


class STTMetrics(TypedDict):
    timestamp: float
    duration: float
    label: str


@unique
class SpeechEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT"""
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking"""


@dataclass
class SpeechData:
    language: str
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0  # [0, 1]


@dataclass
class SpeechEvent:
    type: SpeechEventType
    request_id: str = ""
    alternatives: List[SpeechData] = field(default_factory=list)


@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool


class STT(ABC, rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(self, *, capabilities: STTCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

    @abstractmethod
    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: str | None = None
    ) -> SpeechEvent: ...

    async def recognize(
        self, buffer: AudioBuffer, *, language: str | None = None
    ) -> SpeechEvent:
        start_time = time.perf_counter()
        event = await self._recognize_impl(buffer, language=language)
        duration = time.perf_counter() - start_time
        stt_metrics: STTMetrics = {
            "timestamp": time.time(),
            "duration": duration,
            "label": self._label,
        }
        self.emit("metrics_collected", stt_metrics)
        return event

    def stream(self, *, language: str | None = None) -> "SpeechStream":
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"
        )

    async def aclose(self) -> None:
        """
        Close the STT, and every stream/requests associated with it
        """
        ...


class SpeechStream(ABC):
    class _FlushSentinel:
        pass

    def __init__(self, *, sample_rate: int | None = None):
        """
        Args:
        sample_rate : int or None, optional
            The desired sample rate for the audio input.
            If specified, the audio input will be automatically resampled to match
            the given sample rate before being processed for Speech-to-Text.
            If not provided (None), the input will retain its original sample rate.
        """
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, SpeechStream._FlushSentinel]]()
        self._event_ch = aio.Chan[SpeechEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

        self._needed_sr = sample_rate
        self._pushed_sr = 0
        self._resampler: rtc.AudioResampler | None = None

    @abstractmethod
    async def _main_task(self) -> None: ...

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push audio to be recognized"""
        self._check_input_not_ended()
        self._check_not_closed()

        if self._pushed_sr and self._pushed_sr != frame.sample_rate:
            raise ValueError("the sample rate of the input frames must be consistent")

        self._pushed_sr = frame.sample_rate

        if self._needed_sr and self._needed_sr != frame.sample_rate:
            if not self._resampler:
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    self._needed_sr,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )

        if self._resampler:
            for frame in self._resampler.push(frame):
                self._input_ch.send_nowait(frame)
        else:
            self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._check_input_not_ended()
        self._check_not_closed()

        if self._resampler:
            for frame in self._resampler.flush():
                self._input_ch.send_nowait(frame)

        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark the end of input, no more text will be pushed"""
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        """Close ths stream immediately"""
        self._input_ch.close()
        await aio.gracefully_cancel(self._task)
        self._event_ch.close()

    async def __anext__(self) -> SpeechEvent:
        if self._task.done() and (exc := self._task.exception()):
            raise exc

        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
