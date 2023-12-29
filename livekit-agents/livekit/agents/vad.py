from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass, field
from livekit import rtc
from enum import Enum
import datetime


class VADType(Enum):
    START_SPEAKING = 1
    END_SPEAKING = 2


@dataclass
class VADOptions:
    # Minimum duration of speech to trigger a START_SPEAKING event
    min_speaking_duration: float = 0.5
    # Minimum duration of silence to trigger an END_SPEAKING event
    min_silence_duration: float = 0.5
    # Frames to pad the start and end of the speech with
    padding_duration: float = 0.1


@dataclass
class VADEvent:
    type: VADType
    # Timestamp of the event
    timestamp: datetime.datetime

    # END_SPEAKING fields:
    # Duration of the speech in seconds
    duration: float = 0.0
    # List of audio frames of the speech
    speech: List[rtc.AudioFrame] = field(default_factory=list)


class VAD(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def stream(self, opts: VADOptions) -> "VADStream":
        pass


class VADStream(ABC):
    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> VADEvent:
        raise StopAsyncIteration

    def __aiter__(self) -> "VADStream":
        return self
