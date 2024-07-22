from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, List

from livekit import rtc

from ..utils import AudioBuffer


@unique
class SpeechEventType(Enum):
    START_OF_SPEECH = "start_of_speech"
    """indicate the start of speech
    if the STT doesn't support this event, this will be emitted as the same time as the first INTERIM_TRANSCRIPT"""
    INTERIM_TRANSCRIPT = "interim_transcript"
    """interim transcript, useful for real-time transcription"""
    FINAL_TRANSCRIPT = "final_transcript"
    """final transcript, emitted when the STT is confident enough that a certain
    portion of speech will not change"""
    END_OF_SPEECH = "end_of_speech"
    """indicate the end of speech, emitted when the user stops speaking"""


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


@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool


class STT(ABC):
    def __init__(self, *, capabilities: STTCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> STTCapabilities:
        return self._capabilities

    @abstractmethod
    async def recognize(
        self, buffer: AudioBuffer, *, language: str | None = None
    ) -> SpeechEvent:
        pass

    def stream(self, *, language: str | None = None) -> "SpeechStream":
        raise NotImplementedError(
            "streaming is not supported by this STT, please use \
            a different STT or use a StreamAdapter"
        )

    async def aclose(self) -> None:
        """
        Close the STT, and every stream/requests associated with it
        """
        pass


class SpeechStream(ABC):
    @abstractmethod
    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """
        Push a frame to be recognized, it is recommended to push
        frames as soon as they are available
        """
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """
        Close the stream, if wait is True, it will wait for the STT to finish processing the
        remaining frames, otherwise it will close the stream immediately
        """
        pass

    @abstractmethod
    async def __anext__(self) -> SpeechEvent:
        pass

    def __aiter__(self) -> AsyncIterator[SpeechEvent]:
        return self
