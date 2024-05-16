from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

from livekit import rtc

from ..utils import misc


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


class ChunkedStream(ABC):
    """
    Used by the non-streamed synthesize API, some providers support chunked http responses
    """

    async def collect(self) -> rtc.AudioFrame:
        """
        Utility method to collect every frame in a single call
        """
        frames = []
        async for ev in self:
            frames.append(ev.data)

        return misc.merge_frames(frames)

    @abstractmethod
    async def __anext__(self) -> SynthesizedAudio: ...

    @abstractmethod
    async def aclose(self) -> None:
        """close is automatically called if the stream is completely collected"""
        ...

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self


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
    async def aclose(self, *, wait: bool = True) -> None:
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
    def __init__(
        self, *, streaming_supported: bool, sample_rate: int, num_channels: int
    ) -> None:
        self._streaming_supported = streaming_supported
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @abstractmethod
    def synthesize(self, text: str) -> ChunkedStream: ...

    def stream(self) -> SynthesizeStream:
        raise NotImplementedError(
            "streaming is not supported by this TTS, please use a different TTS or use a StreamAdapter"
        )

    @property
    def streaming_supported(self) -> bool:
        return self._streaming_supported
