from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, List

from livekit import rtc


@unique
class VADEventType(Enum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class VADEvent:
    type: VADEventType
    """type of the event"""
    samples_index: int
    """index of the samples when the event was fired"""
    duration: float
    """duration of the speech in seconds"""
    frames: List[rtc.AudioFrame] = field(default_factory=list)
    """list of audio frames of the speech"""
    probability: float = 0.0
    """smoothed probability of the speech (only for INFERENCE_DONE event)"""
    inference_duration: float = 0.0
    """duration of the inference in seconds (only for INFERENCE_DONE event)"""
    speaking: bool = False
    """whether speech was detected in the frames"""


class VAD(ABC):
    def __init__(self, *, update_interval: float) -> None:
        self._update_interval = update_interval

    @property
    def update_interval(self) -> float:
        """interval in seconds to update the VAD model"""
        return self._update_interval

    @abstractmethod
    def stream(
        self,
    ) -> "VADStream":
        pass


class VADStream(ABC):
    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> VADEvent: ...

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[VADEvent]: ...
