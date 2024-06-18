from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, List

from livekit import rtc


class VADEventType(Enum):
    START_OF_SPEECH = 1
    INFERENCE_DONE = 2
    END_OF_SPEECH = 3


@dataclass
class VADEvent:
    type: VADEventType
    """type of the event"""
    samples_index: int
    """index of the samples of the event (when the event was fired)"""
    duration: float = 0.0
    """duration of the speech in seconds (only for END_SPEAKING event)"""
    frames: List[rtc.AudioFrame] = field(default_factory=list)
    """list of audio frames of the speech"""
    probability: float = 0.0
    """smoothed probability of the speech (only for INFERENCE_DONE event)"""
    inference_duration: float = 0.0
    """duration of the inference in seconds (only for INFERENCE_DONE event)"""
    speaking: bool = False
    """whether speech was detected in the frames"""


class VAD(ABC):
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
