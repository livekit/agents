from abc import ABC, abstractmethod
from livekit import rtc
from typing import List
from dataclasses import dataclass
from .utils import AudioBuffer


@dataclass
class SpeechData:
    language: str
    start_time: float
    end_time: float
    confidence: float  # [0, 1]
    text: str


@dataclass
class SpeechEvent:
    is_final: bool
    alternatives: List[SpeechData]


@dataclass
class RecognizeOptions:
    language: str = "en-US"
    detect_language: bool = False
    num_channels: int = 1
    sample_rate: int = 16000


@dataclass
class StreamOptions:
    language: str = "en-US"
    detect_language: bool = False
    interim_results: bool = True
    num_channels: int = 1
    sample_rate: int = 16000  # sane default for STT


class SpeechStream(ABC):
    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        pass

    @abstractmethod
    async def flush(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    def __aiter__(self) -> "SpeechStream":
        return self

    async def __anext__(self) -> SpeechEvent:
        raise NotImplementedError


class STT(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def recognize(
        self,
        buffer: AudioBuffer,
        opts: RecognizeOptions = RecognizeOptions(),
    ) -> SpeechEvent:
        pass

    @abstractmethod
    def stream(self, opts: StreamOptions = StreamOptions()) -> SpeechStream:
        pass
