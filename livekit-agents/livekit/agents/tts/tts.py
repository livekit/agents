from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
from livekit import rtc
from ..utils import misc


@dataclass
class SynthesizedAudio:
    segment_id: str
    """Segment ID is used to identify the same synthesized audio across different SynthesizedAudio events"""
    frame: rtc.AudioFrame
    """Synthesized audio frame"""
    delta_text: str = ""
    """Current segment of the synthesized audio"""
    end_of_segment: bool = False
    """True if this is the last frame in the segment"""


class ChunkedStream(ABC):
    """Used by the non-streamed synthesize API, some providers support chunked http responses"""

    async def collect(self) -> rtc.AudioFrame:
        """Utility method to collect every frame in a single call"""
        frames = []
        async for ev in self:
            frames.append(ev.frame)
        return misc.merge_frames(frames)

    @abstractmethod
    async def aclose(self) -> None:
        """Close is automatically called if the stream is completely collected"""
        ...

    @abstractmethod
    async def __anext__(self) -> SynthesizedAudio: ...

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self


class SynthesizeStream(ABC):
    @abstractmethod
    def push_text(self, token: str | None) -> None:
        """Push some text to be synthesized. If token is None,
        it will be used to identify the end of this particular segment."""
        pass

    def flush(self) -> None:
        """Mark the end of the current segment, this is equivalent to calling
        push_text(None)"""
        self.push_text(None)

    @abstractmethod
    async def aclose(self) -> None:
        """Close ths stream immediately"""
        ...

    @abstractmethod
    async def __anext__(self) -> SynthesizedAudio:
        pass

    def __aiter__(self) -> AsyncIterator[SynthesizedAudio]:
        return self


@dataclass
class TTSCapabilities:
    streaming: bool


class TTS(ABC):
    def __init__(
        self, *, capabilities: TTSCapabilities, sample_rate: int, num_channels: int
    ) -> None:
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    @property
    def capabilities(self) -> TTSCapabilities:
        return self._capabilities

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

    async def aclose(self) -> None: ...
