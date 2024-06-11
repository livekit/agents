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
    raw_inference_prob: float = 0.0
    """raw probability of the speech (only for INFERENCE_DONE event)"""
    inference_duration: float = 0.0
    """duration of the inference in seconds (only for INFERENCE_DONE event)"""
    speaking: bool = False
    """whether speech was detected in the frames"""


class VAD(ABC):
    @abstractmethod
    def stream(
        self,
        *,
        min_speaking_duration: float = 0.16,
        min_silence_duration: float = 1.3,
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
    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        pass

    @abstractmethod
    async def aclose(self, *, wait: bool = True) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> VADEvent:
        raise StopAsyncIteration

    def __aiter__(self) -> AsyncIterator[VADEvent]:
        return self
