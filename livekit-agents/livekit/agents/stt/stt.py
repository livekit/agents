from abc import ABC, abstractmethod
from livekit import rtc
from typing import Optional, List
from dataclasses import dataclass
from ..utils import AudioBuffer
from enum import Enum

class SpeechEventType(Enum):
    START_OF_SPEECH = 0
    INTERIM_TRANSCRIPT = 1
    FINAL_TRANSCRIPT = 2
    END_OF_SPEECH = 3


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
    # When the type is END_OF_SPEECH, the first alternative will be a combination of all the received final transcripts
    alternatives: List[SpeechData]


class STT(ABC):
    def __init__(
        self,
        *,
        streaming_supported: bool,
    ) -> None:
        self._streaming_supported = streaming_supported

    @abstractmethod
    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[str] = None,
    ) -> SpeechEvent:
        pass

    def stream(
        self,
        *,
        language: Optional[str] = None,
    ) -> "SpeechStream":
        raise NotImplementedError(
            "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"
        )

    @property
    def streaming_supported(self) -> bool:
        return self._streaming_supported


class SpeechStream(ABC):
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
    async def __anext__(self) -> SpeechEvent:
        pass

    def __aiter__(self) -> "SpeechStream":
        return self
