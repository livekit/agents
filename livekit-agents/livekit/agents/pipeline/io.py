from __future__ import annotations

import asyncio
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
from ..log import logger

STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame]],
    Union[Awaitable[Optional[AsyncIterable[stt.SpeechEvent]]]],  # TODO: support str
]
LLMNode = Callable[
    [llm.ChatContext, list[llm.AIFunction]],
    Union[
        Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        Awaitable[Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],],
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


class TextSink(ABC):
    @abstractmethod
    async def capture_text(self, text: str) -> None:
        """Capture a text segment (Used by the output of LLM nodes)"""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Mark the current text segment as complete (e.g LLM generation is complete)"""
        ...


class ParallelTextSink(TextSink):
    def __init__(self, *sinks: TextSink) -> None:
        self._sinks = sinks

    async def capture_text(self, text: str) -> None:
        await asyncio.gather(*[sink.capture_text(text) for sink in self._sinks])

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()


# TODO(theomonnom): Add documentation to VideoSink
class VideoSink(ABC):
    @abstractmethod
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class AgentInput:
    def __init__(self, video_changed: Callable, audio_changed: Callable) -> None:
        self._video_stream: VideoStream | None = None
        self._audio_stream: AudioStream | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed

        self._paused_video_stream: VideoStream | None = None
        self._paused_audio_stream: AudioStream | None = None
        self._task_consume_audio: asyncio.Task | None = None
        self._task_consume_video: asyncio.Task | None = None

    def toggle_audio(self, enable: bool) -> None:
        if enable:
            if self._audio_stream is None and self._paused_audio_stream is not None:
                self.audio = self._paused_audio_stream
                logger.debug("input audio stream resumed")
            elif self._paused_audio_stream is None:
                logger.warning("no audio stream to resume")
        else:
            if self._audio_stream is not None:
                self._paused_audio_stream = self._audio_stream
                self.audio = None
                if self._task_consume_audio:
                    self._task_consume_audio.cancel()
                self._task_consume_audio = asyncio.create_task(
                    self._consume_stream(self._paused_audio_stream)
                )
                logger.debug("input audio stream paused")

    def toggle_video(self, enable: bool) -> None:
        if enable:
            if self._video_stream is None and self._paused_video_stream is not None:
                self.video = self._paused_video_stream
                self._paused_video_stream = None
                logger.debug("input video stream resumed")
            elif self._paused_video_stream is None:
                logger.warning("no video stream to resume")
        else:
            if self._video_stream is not None:
                self._paused_video_stream = self._video_stream
                self.video = None
                if self._task_consume_video:
                    self._task_consume_video.cancel()
                self._task_consume_video = asyncio.create_task(
                    self._consume_stream(self._paused_video_stream)
                )
                logger.debug("input video stream paused")

    @property
    def video(self) -> VideoStream | None:
        return self._video_stream

    @video.setter
    def video(self, stream: VideoStream | None) -> None:
        self._video_stream = stream
        if stream is not None:
            self._paused_video_stream = None
            if self._task_consume_video:
                self._task_consume_video.cancel()
                self._task_consume_video = None
        self._video_changed()

    @property
    def audio(self) -> AudioStream | None:
        return self._audio_stream

    @audio.setter
    def audio(self, stream: AudioStream | None) -> None:
        self._audio_stream = stream
        if stream is not None:
            self._paused_audio_stream = None
            if self._task_consume_audio:
                self._task_consume_audio.cancel()
                self._task_consume_audio = None
        self._audio_changed()

    async def _consume_stream(self, stream: AudioStream | VideoStream) -> None:
        async for frame in stream:
            pass


class AgentOutput:
    def __init__(
        self, video_changed: Callable, audio_changed: Callable, text_changed: Callable
    ) -> None:
        self._video_sink: VideoSink | None = None
        self._audio_sink: AudioSink | None = None
        self._text_sink: TextSink | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._text_changed = text_changed

        self._paused_video_sink: VideoSink | None = None
        self._paused_audio_sink: AudioSink | None = None
        self._paused_text_sink: TextSink | None = None

    def toggle_video(self, enable: bool) -> None:
        if enable:
            if self._video_sink is None and self._paused_video_sink is not None:
                self.video = self._paused_video_sink
                self._paused_video_sink = None
                logger.debug("output video sink resumed")
            elif self._paused_video_sink is None:
                logger.warning("no video sink to resume")
        else:
            if self._video_sink is not None:
                self._paused_video_sink = self._video_sink
                self.video = None
                logger.debug("output video sink paused")

    def toggle_audio(self, enable: bool) -> None:
        if enable:
            if self._audio_sink is None and self._paused_audio_sink is not None:
                self.audio = self._paused_audio_sink
                self._paused_audio_sink = None
                logger.debug("output audio sink resumed")
            elif self._paused_audio_sink is None:
                logger.warning("no audio sink to resume")
        else:
            if self._audio_sink is not None:
                self._paused_audio_sink = self._audio_sink
                self.audio = None
                logger.debug("output audio sink paused")

    def toggle_text(self, enable: bool) -> None:
        if enable:
            if self._text_sink is None and self._paused_text_sink is not None:
                self.text = self._paused_text_sink
                self._paused_text_sink = None
                logger.debug("output text sink resumed")
            elif self._paused_text_sink is None:
                logger.warning("no text sink to resume")
        else:
            if self._text_sink is not None:
                self._paused_text_sink = self._text_sink
                self.text = None
                logger.debug("output text sink paused")

    @property
    def video(self) -> VideoSink | None:
        return self._video_sink

    @video.setter
    def video(self, sink: VideoSink | None) -> None:
        self._video_sink = sink
        self._video_changed()

    @property
    def audio(self) -> AudioSink | None:
        return self._audio_sink

    @audio.setter
    def audio(self, sink: AudioSink | None) -> None:
        self._audio_sink = sink
        self._audio_changed()

    @property
    def text(self) -> TextSink | None:
        return self._text_sink

    @text.setter
    def text(self, sink: TextSink | None) -> None:
        self._text_sink = sink
        self._text_changed()
