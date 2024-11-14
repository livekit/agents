from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterable, AsyncIterator, List, Literal, Union

from livekit import rtc

from .metrics import VADMetrics
from .utils import aio


@unique
class VADEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class VADEvent:
    """
    Represents an event detected by the Voice Activity Detector (VAD).
    """

    type: VADEventType
    """Type of the VAD event (e.g., start of speech, end of speech, inference done)."""

    samples_index: int
    """Index of the audio sample where the event occurred, relative to the inference sample rate."""

    timestamp: float
    """Timestamp (in seconds) when the event was fired."""

    speech_duration: float
    """Duration of the speech segment in seconds."""

    silence_duration: float
    """Duration of the silence segment in seconds."""

    frames: List[rtc.AudioFrame] = field(default_factory=list)
    """
    List of audio frames associated with the speech.

    - For `start_of_speech` events, this contains the audio chunks that triggered the detection.
    - For `inference_done` events, this contains the audio chunks that were processed.
    - For `end_of_speech` events, this contains the complete user speech.
    """

    probability: float = 0.0
    """Probability that speech is present (only for `INFERENCE_DONE` events)."""

    inference_duration: float = 0.0
    """Time taken to perform the inference, in seconds (only for `INFERENCE_DONE` events)."""

    speaking: bool = False
    """Indicates whether speech was detected in the frames."""

    raw_accumulated_silence: float = 0.0
    """Threshold used to detect silence."""

    raw_accumulated_speech: float = 0.0
    """Threshold used to detect speech."""


@dataclass
class VADCapabilities:
    update_interval: float


class VAD(ABC, rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(self, *, capabilities: VADCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def capabilities(self) -> VADCapabilities:
        return self._capabilities

    @abstractmethod
    def stream(self) -> "VADStream": ...


class VADStream(ABC):
    class _FlushSentinel:
        pass

    def __init__(self, vad: VAD) -> None:
        self._vad = vad
        self._last_activity_time = time.perf_counter()
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, VADStream._FlushSentinel]]()
        self._event_ch = aio.Chan[VADEvent]()

        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="TTS._metrics_task"
        )

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _main_task(self) -> None: ...

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[VADEvent]) -> None:
        """Task used to collect metrics"""

        inference_duration_total = 0.0
        inference_count = 0

        async for ev in event_aiter:
            if ev.type == VADEventType.INFERENCE_DONE:
                inference_duration_total += ev.inference_duration
                inference_count += 1

                if inference_count >= 1 / self._vad.capabilities.update_interval:
                    vad_metrics = VADMetrics(
                        timestamp=time.time(),
                        idle_time=time.perf_counter() - self._last_activity_time,
                        inference_duration_total=inference_duration_total,
                        inference_count=inference_count,
                        label=self._vad._label,
                    )
                    self._vad.emit("metrics_collected", vad_metrics)

                    inference_duration_total = 0.0
                    inference_count = 0
            elif ev.type in [VADEventType.START_OF_SPEECH, VADEventType.END_OF_SPEECH]:
                self._last_activity_time = time.perf_counter()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push some text to be synthesized"""
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(frame)

    def flush(self) -> None:
        """Mark the end of the current segment"""
        self._check_input_not_ended()
        self._check_not_closed()
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
        await self._metrics_task

    async def __anext__(self) -> VADEvent:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if self._task.done() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[VADEvent]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")
