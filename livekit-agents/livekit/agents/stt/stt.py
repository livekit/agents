from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from livekit import rtc

from ..utils import AudioBuffer


class SpeechEventType(Enum):
    # indicate the start of speech
    # if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT
    START_OF_SPEECH = 0
    # interim transcript, useful for real-time transcription
    INTERIM_TRANSCRIPT = 1
    # final transcript, emitted when the STT is confident enough that a certain
    # portion of speech will not change
    FINAL_TRANSCRIPT = 2
    # indicate the end of speech, emitted when the user stops speaking
    # the first alternative is a combination of all the previous FINAL_TRANSCRIPT events
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
    alternatives: List[SpeechData] = field(default_factory=list)


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
        language: str | None = None,
    ) -> SpeechEvent:
        pass

    def stream(
        self,
        *,
        language: str | None = None,
    ) -> "SpeechStream":
        raise NotImplementedError(
            "streaming is not supported by this STT, please use \
            a different STT or use a StreamAdapter"
        )

    @property
    def streaming_supported(self) -> bool:
        return self._streaming_supported


class SpeechStream(ABC):
    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Push a frame to be recognized, it is recommended to push
        frames as soon as they are available
        """
        pass

    @abstractmethod
    async def aclose(self, *, wait: bool = True) -> None:
        """
        Close the stream, if wait is True, it will wait for the STT to finish processing the
        remaining frames, otherwise it will close the stream immediately
        """
        pass

    @abstractmethod
    async def __anext__(self) -> SpeechEvent:
        pass

    def __aiter__(self) -> "SpeechStream":
        return self
