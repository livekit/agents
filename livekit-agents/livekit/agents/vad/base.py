from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, List, Union

from livekit import rtc

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
    """Duration of the detected speech segment in seconds."""

    silence_duration: float
    """Duration of the silence segment preceding or following the speech, in seconds."""

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


@dataclass
class VADCapabilities:
    update_interval: float


class VAD(ABC):
    def __init__(self, *, capabilities: VADCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> VADCapabilities:
        return self._capabilities

    @abstractmethod
    def stream(self) -> "VADStream":
        pass


class VADStream(ABC):
    class _FlushSentinel:
        pass

    def __init__(self):
        self._input_ch = aio.Chan[Union[rtc.AudioFrame, VADStream._FlushSentinel]]()
        self._event_ch = aio.Chan[VADEvent]()
        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _main_task(self) -> None: ...

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

    async def __anext__(self) -> VADEvent:
        return await self._event_ch.__anext__()

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
