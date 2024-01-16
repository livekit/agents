from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass, field
from livekit import rtc
from enum import Enum


class VADEventType(Enum):
    START_SPEAKING = 1
    SPEAKING = 2
    END_SPEAKING = 3


@dataclass
class VADEvent:
    # type of the event
    type: VADEventType
    # index of the samples of the event (when the event was fired)
    samples_index: int
    # duration of the speech in seconds (only for END_SPEAKING event)
    duration: float = 0.0
    # list of audio frames of the speech
    speech: List[rtc.AudioFrame] = field(default_factory=list)


class VAD(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def stream(
        self,
        *,
        min_speaking_duration: float = 0.5,
        min_silence_duration: float = 0.5,
        padding_duration: float = 0.1,
        sample_rate: int = 16000,
        max_buffered_speech: float = 45.0,
    ) -> "VADStream":
        """Returns a VADStream that can be used to push audio frames and receive VAD events
        Args:
          min_speaking_duration: minimum duration of speech to trigger a START_SPEAKING event

          min_silence_duration: in the end of each speech chunk wait for min_silence_duration_ms before separating it
              the padding is not always precise, it is generally rounded to the nearest 40ms depending on the vad implementation

          padding_duration: frames to pad the start and end of the speech with

          sample_rate: sample rate of the inference/processing

          max_buffered_speech: max buffered speech in seconds that we keep until the END_SPEAKING event is triggered
              it is unrecommended to use 0.0 as it may cause OOM if the user doesn't stop speaking"""
        pass


class VADStream(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        pass

    @abstractmethod
    async def flush(self) -> None:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> VADEvent:
        raise StopAsyncIteration

    def __aiter__(self) -> "VADStream":
        return self
