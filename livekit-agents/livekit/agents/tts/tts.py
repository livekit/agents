from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterable, Optional

from livekit import rtc


@dataclass
class SynthesizedAudio:
    text: str
    data: rtc.AudioFrame


class SynthesisEventType(Enum):
    # first event, indicates that the stream has started
    # retriggered after FINISHED
    STARTED = 0
    # audio data is available
    AUDIO = 1
    # generally happens after a flushing the stream
    # some TTS providers close the stream so we know it's done
    FINISHED = 2


@dataclass
class SynthesisEvent:
    type: SynthesisEventType
    audio: Optional[SynthesizedAudio] = None


class SynthesizeStream(ABC):
    @abstractmethod
    def push_text(self, token: str) -> None:
        pass

    @abstractmethod
    async def flush(self) -> None:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> SynthesisEvent:
        pass

    def __aiter__(self) -> "SynthesizeStream":
        return self


class TTS(ABC):
    def __init__(self, *, streaming_supported: bool) -> None:
        self._streaming_supported = streaming_supported

    @abstractmethod
    def synthesize(self, *, text: str) -> AsyncIterable[SynthesizedAudio]:
        pass

    def stream(self) -> SynthesizeStream:
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )

    @property
    def streaming_supported(self) -> bool:
        return self._streaming_supported
