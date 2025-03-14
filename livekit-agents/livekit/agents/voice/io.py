from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    AsyncIterable,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Union,
)

from livekit import rtc

from .. import llm, stt
from ..log import logger

# TODO(theomonnom): can those types be simplified?
STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame]],
    Union[
        Optional[Union[AsyncIterable[stt.SpeechEvent], AsyncIterable[str]]],
        Awaitable[Optional[Union[AsyncIterable[stt.SpeechEvent], AsyncIterable[str]]]],
    ],
]
LLMNode = Callable[
    [llm.ChatContext, list[llm.FunctionTool]],
    Union[
        Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        Awaitable[Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]]],
    ],
]
TTSNode = Callable[
    [AsyncIterable[str]],
    Union[
        Optional[AsyncIterable[rtc.AudioFrame]],
        Awaitable[Optional[AsyncIterable[rtc.AudioFrame]]],
    ],
]


class AudioInput:
    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame: ...

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


class VideoInput:
    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame]:
        return self

    async def __anext__(self) -> rtc.VideoFrame: ...

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


@dataclass
class PlaybackFinishedEvent:
    playback_position: float
    """How much of the audio was played back"""
    interrupted: bool
    """interrupted is True if playback was interrupted (clear_buffer() was called)"""


class AudioOutput(ABC, rtc.EventEmitter[Literal["playback_finished"]]):
    def __init__(self, *, sample_rate: int | None = None) -> None:
        """
        Args:
            sample_rate: The sample rate required by the audio sink, if None, any sample rate is accepted
        """
        super().__init__()
        self._sample_rate = sample_rate
        self.__capturing = False
        self.__playback_finished_event = asyncio.Event()

        self.__playback_segments_count = 0
        self.__playback_finished_count = 0
        self.__last_playback_ev: PlaybackFinishedEvent = PlaybackFinishedEvent(
            playback_position=0, interrupted=False
        )

    def on_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        """
        Developers building audio sinks must call this method when a playback/segment is finished.
        Segments are segmented by calls to flush() or clear_buffer()
        """

        if self.__playback_finished_count >= self.__playback_segments_count:
            logger.warning(
                "playback_finished called more times than playback segments were captured"
            )
            return

        self.__playback_finished_count += 1
        self.__playback_finished_event.set()

        ev = PlaybackFinishedEvent(playback_position=playback_position, interrupted=interrupted)
        self.__last_playback_ev = ev
        self.emit("playback_finished", ev)

    async def wait_for_playout(self) -> PlaybackFinishedEvent:
        """
        Wait for the past audio segments to finish playing out.

        Returns:
            PlaybackFinishedEvent: The event that was emitted when the audio finished playing out
            (only the last segment information)
        """
        target = self.__playback_segments_count

        while self.__playback_finished_count < target:
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
            self.__playback_segments_count += 1

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered audio, marking the current playback/segment as complete"""
        self.__capturing = False

    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the buffer, stopping playback immediately"""

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


class TextOutput(ABC):
    @abstractmethod
    async def capture_text(self, text: str) -> None:
        """Capture a text segment (Used by the output of LLM nodes)"""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Mark the current text segment as complete (e.g LLM generation is complete)"""
        ...

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


# TODO(theomonnom): Add documentation to VideoSink
class VideoOutput(ABC):
    @abstractmethod
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


class AgentInput:
    def __init__(self, video_changed: Callable, audio_changed: Callable) -> None:
        self._video_stream: VideoInput | None = None
        self._audio_stream: AudioInput | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed

        self._audio_enabled = False
        self._video_enabled = False

    def set_audio_enabled(self, enable: bool):
        if enable == self._audio_enabled:
            return

        self._audio_enabled = enable

        if not self._audio_stream:
            return

        if enable:
            self._audio_stream.on_attached()
        else:
            self._audio_stream.on_detached()

    def set_video_enabled(self, enable: bool):
        if enable == self._video_enabled:
            return

        self._video_enabled = enable

        if not self._video_stream:
            return

        if enable:
            self._video_stream.on_attached()
        else:
            self._video_stream.on_detached()

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def video_enabled(self) -> bool:
        return self._video_enabled

    @property
    def video(self) -> VideoInput | None:
        return self._video_stream

    @video.setter
    def video(self, stream: VideoInput | None) -> None:
        self._video_stream = stream
        self._video_changed()

    @property
    def audio(self) -> AudioInput | None:
        return self._audio_stream

    @audio.setter
    def audio(self, stream: AudioInput | None) -> None:
        self._audio_stream = stream
        self._audio_changed()


class AgentOutput:
    def __init__(
        self, video_changed: Callable, audio_changed: Callable, transcription_changed: Callable
    ) -> None:
        self._video_sink: VideoOutput | None = None
        self._audio_sink: AudioOutput | None = None
        self._transcription_sink: TextOutput | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._transcription_changed = transcription_changed

        self._audio_enabled = False
        self._video_enabled = False
        self._transcription_enabled = False

    def set_video_enabled(self, enabled: bool):
        if enabled == self._video_enabled:
            return

        self._video_enabled = enabled

        if not self._video_sink:
            return

        if enabled:
            self._video_sink.on_attached()
        else:
            self._video_sink.on_detached()

    def set_audio_enabled(self, enabled: bool):
        if enabled == self._audio_enabled:
            return

        self._audio_enabled = enabled

        if not self._audio_sink:
            return

        if enabled:
            self._audio_sink.on_attached()
        else:
            self._audio_sink.on_detached()

    def set_transcription_enabled(self, enabled: bool):
        if enabled == self._transcription_enabled:
            return

        self._transcription_enabled = enabled

        if not self._transcription_sink:
            return

        if enabled:
            self._transcription_sink.on_attached()
        else:
            self._transcription_sink.on_detached()

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def video_enabled(self) -> bool:
        return self._video_enabled

    @property
    def transcription_enabled(self) -> bool:
        return self._transcription_enabled

    @property
    def video(self) -> VideoOutput | None:
        return self._video_sink

    @video.setter
    def video(self, sink: VideoOutput | None) -> None:
        self._video_sink = sink
        self._video_changed()

    @property
    def audio(self) -> AudioOutput | None:
        return self._audio_sink

    @audio.setter
    def audio(self, sink: AudioOutput | None) -> None:
        self._audio_sink = sink
        self._audio_changed()

    @property
    def transcription(self) -> TextOutput | None:
        return self._transcription_sink

    @transcription.setter
    def transcription(self, sink: TextOutput | None) -> None:
        self._transcription_sink = sink
        self._transcription_changed()
