from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterable

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
    # finished synthesizing an audio segment (generally separated by sending "None" to push_text)
    # this doesn't means the stream is done, more text can be pushed
    FINISHED = 2


@dataclass
class SynthesisEvent:
    type: SynthesisEventType
    audio: SynthesizedAudio | None = None


class SynthesizeStream(ABC):
    @abstractmethod
    def push_text(self, token: str | None) -> None:
        """
        Push some text to be synthesized. If token is None,
        it will be used to identify the end of this particular segment.
        (required by some TTS engines)
        """
        pass

    def mark_segment_end(self) -> None:
        """
        Mark the end of the current segment, this is equivalent to calling
        push_text(None)
        """
        self.push_text(None)

    @abstractmethod
    async def aclose(self, wait: bool = True) -> None:
        """
        Close the stream, if wait is True, it will wait for the TTS to
        finish synthesizing the audio, otherwise it will close ths stream immediately
        """
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
    def synthesize(self, text: str) -> AsyncIterable[SynthesizedAudio]:
        raise NotImplementedError("synthesize is not implemented")

    def stream(self) -> SynthesizeStream:
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )

    @property
    def streaming_supported(self) -> bool:
        return self._streaming_supported
