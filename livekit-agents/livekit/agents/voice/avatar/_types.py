from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import Generic, Literal, TypeVar, Union

from livekit import rtc


class AudioSegmentEnd:
    pass


TEvent = TypeVar("TEvent")


class AudioReceiver(
    ABC,
    rtc.EventEmitter[Union[Literal["clear_buffer", "pause", "resume"], TEvent]],
    Generic[TEvent],
):
    async def start(self) -> None:
        pass

    @abstractmethod
    def notify_playback_finished(
        self, playback_position: float, interrupted: bool
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has finished"""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        """Continuously stream out audio frames or AudioSegmentEnd when the stream ends"""

    async def aclose(self) -> None:
        pass


class VideoGenerator(ABC):
    @abstractmethod
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Push an audio frame to the video generator"""

    @abstractmethod
    def clear_buffer(self) -> None | Coroutine[None, None, None]:
        """Clear the audio buffer, stopping audio playback immediately"""

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """Continuously stream out video and audio frames, or AudioSegmentEnd when the audio segment ends"""  # noqa: E501

    def pause(self) -> None:  # noqa: B027
        """Pause the audio output temporarily"""

    def resume(self) -> None:  # noqa: B027
        """Resume the audio output"""
