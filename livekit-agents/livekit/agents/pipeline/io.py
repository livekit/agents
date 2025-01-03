from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    AsyncIterable,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Union,
)

from livekit import rtc

from .. import llm, stt

import asyncio

STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame]],
    Union[Awaitable[Optional[AsyncIterable[stt.SpeechEvent]]],],
]
LLMNode = Callable[
    [llm.ChatContext, Optional[llm.FunctionContext]],
    Union[
        Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        Awaitable[
            Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        ],
    ],
]
TTSNode = Callable[
    [AsyncIterable[str]],
    Union[
        Optional[AsyncIterable[rtc.AudioFrame]],
        Awaitable[Optional[AsyncIterable[rtc.AudioFrame]]],
    ],
]


AudioStream = AsyncIterable[rtc.AudioFrame]
VideoStream = AsyncIterable[rtc.VideoFrame]


@dataclass
class PlaybackFinishedEvent:
    playback_position: float
    """How much of the audio was played back"""
    interrupted: bool
    """interrupted is True if playback was interrupted (clear_buffer() was called)"""


class AudioSink(ABC, rtc.EventEmitter[Literal["playback_finished"]]):
    def __init__(self, *, sample_rate: int | None = None) -> None:
        """
        Args:
            sample_rate: The sample rate required by the audio sink, if None, any sample rate is accepted
        """
        super().__init__()
        self._sample_rate = sample_rate
        self.__capturing = False
        self.__playback_finished_event = asyncio.Event()

        self.__nb_playback_finished_needed = 0
        self.__playback_finished_count = 0

    def on_playback_finished(
        self, *, playback_position: float, interrupted: bool
    ) -> None:
        """
        Developers building audio sinks must call this method when a playback/segment is finished.
        Segments are segmented by calls to flush() or clear_buffer()
        """
        self.__nb_playback_finished_needed = max(
            0, self.__nb_playback_finished_needed - 1
        )
        self.__playback_finished_count += 1
        self.__playback_finished_event.set()

        ev = PlaybackFinishedEvent(
            playback_position=playback_position, interrupted=interrupted
        )
        self.__last_playback_ev = ev
        self.emit("playback_finished", ev)

    async def wait_for_playout(self) -> PlaybackFinishedEvent:
        """
        Wait for the past audio segments to finish playing out.

        Returns:
            PlaybackFinishedEvent: The event that was emitted when the audio finished playing out
            (only the last segment information)
        """
        needed = self.__nb_playback_finished_needed
        initial_count = self.__playback_finished_count
        target_count = initial_count + needed

        while self.__playback_finished_count < target_count:
            await self.__playback_finished_event.wait()
            self.__playback_finished_event.clear()

        return self.__last_playback_ev

    @property
    def sample_rate(self) -> int | None:
        """The sample rate required by the audio sink, if None, any sample rate is accepted"""
        return self._sample_rate

    @abstractmethod
    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture an audio frame for playback, frames can be pushed faster than real-time"""
        if not self.__capturing:
            self.__capturing = True
            self.__nb_playback_finished_needed += 1

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered audio, marking the current playback/segment as complete"""
        if self.__capturing:
            self.__capturing = False

    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the buffer, stopping playback immediately"""
        ...


class TextSink(ABC):
    @abstractmethod
    async def capture_text(self, text: str) -> None:
        """Capture a text segment (Used by the output of LLM nodes)"""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Mark the current text segment as complete (e.g LLM generation is complete)"""
        ...


# TODO(theomonnom): Add documentation to VideoSink
class VideoSink(ABC):
    @abstractmethod
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...
