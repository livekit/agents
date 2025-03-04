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
TranscriptionNode = Callable[
    [AsyncIterable[str]],
    Union[
        Optional[AsyncIterable[str]],
        Awaitable[Optional[AsyncIterable[str]]],
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

        self._inactive_video_stream: VideoStream | None = None
        self._inactive_audio_stream: AudioStream | None = None
        self._consume_video_task: asyncio.Task | None = None
        self._consume_audio_task: asyncio.Task | None = None

    def set_audio_enabled(self, enable: bool) -> bool:
        if enable and not self._audio_stream and self._inactive_audio_stream:
            self.audio = self._inactive_audio_stream
            return True

        if not enable and self._audio_stream:
            self._inactive_audio_stream = self._audio_stream
            self.audio = None

            # consume the stream in the background
            if self._consume_audio_task:
                self._consume_audio_task.cancel()
            self._consume_audio_task = asyncio.create_task(
                self._consume_stream(self._inactive_audio_stream)
            )
            return True

        return False

    def set_video_enabled(self, enable: bool) -> bool:
        if enable and not self._video_stream and self._inactive_video_stream:
            self.video = self._inactive_video_stream
            return True

        if not enable and self._video_stream:
            self._inactive_video_stream = self._video_stream
            self.video = None

            # consume the stream in the background
            if self._consume_video_task:
                self._consume_video_task.cancel()
            self._consume_video_task = asyncio.create_task(
                self._consume_stream(self._inactive_video_stream)
            )
            return True

        return False

    @property
    def video(self) -> VideoStream | None:
        return self._video_stream

    @video.setter
    def video(self, stream: VideoStream | None) -> None:
        self._video_stream = stream
        if stream is not None:
            # reset the inactive stream
            self._inactive_video_stream = None
            if self._consume_video_task:
                self._consume_video_task.cancel()
                self._consume_video_task = None
        self._video_changed()

    @property
    def audio(self) -> AudioStream | None:
        return self._audio_stream

    @audio.setter
    def audio(self, stream: AudioStream | None) -> None:
        self._audio_stream = stream
        if stream is not None:
            # reset the inactive stream
            self._inactive_audio_stream = None
            if self._consume_audio_task:
                self._consume_audio_task.cancel()
                self._consume_audio_task = None
        self._audio_changed()

    async def _consume_stream(self, stream: AudioStream | VideoStream) -> None:
        async for frame in stream:
            pass

    def close(self) -> None:
        if self._consume_audio_task:
            self._consume_audio_task.cancel()
        if self._consume_video_task:
            self._consume_video_task.cancel()

        self._consume_audio_task = None
        self._consume_video_task = None


class AgentOutput(
    rtc.EventEmitter[Literal["video_changed", "audio_changed", "transcription_changed"]]
):
    def __init__(
        self, video_changed: Callable, audio_changed: Callable, transcription_changed: Callable
    ) -> None:
        super().__init__()
        self._video_sink: VideoSink | None = None
        self._audio_sink: AudioSink | None = None
        self._transcription_sink: TextSink | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._transcription_changed = transcription_changed

        # used to pause/resume streams
        self._inactive_video_sink: VideoSink | None = None
        self._inactive_audio_sink: AudioSink | None = None
        self._inactive_tr_sink: TextSink | None = None

    def set_video_enabled(self, enable: bool) -> bool:
        if enable and not self._video_sink and self._inactive_video_sink:
            self.video = self._inactive_video_sink
            return True

        if not enable and self._video_sink:
            self._inactive_video_sink = self._video_sink
            self.video = None
            return True

        return False

    def set_audio_enabled(self, enable: bool) -> bool:
        if enable and not self._audio_sink and self._inactive_audio_sink:
            self.audio = self._inactive_audio_sink
            return True

        if not enable and self._audio_sink:
            self._inactive_audio_sink = self._audio_sink
            self.audio = None
            return True

        return False

    def set_transcription_enabled(self, enable: bool) -> bool:
        if enable and not self._transcription_sink and self._inactive_tr_sink:
            self.transcription = self._inactive_tr_sink
            return True

        if not enable and self._transcription_sink:
            self._inactive_tr_sink = self._transcription_sink
            self.transcription = None
            return True

        return False

    @property
    def video(self) -> VideoSink | None:
        return self._video_sink

    @video.setter
    def video(self, sink: VideoSink | None) -> None:
        self._video_sink = sink
        if sink is not None:
            self._inactive_video_sink = None
        self.emit("video_changed", sink)
        self._video_changed()

    @property
    def audio(self) -> AudioSink | None:
        return self._audio_sink

    @audio.setter
    def audio(self, sink: AudioSink | None) -> None:
        self._audio_sink = sink
        if sink is not None:
            self._inactive_audio_sink = None
        self.emit("audio_changed", sink)
        self._audio_changed()

    @property
    def transcription(self) -> TextSink | None:
        return self._transcription_sink

    @transcription.setter
    def transcription(self, sink: TextSink | None) -> None:
        self._transcription_sink = sink
        if sink is not None:
            self._inactive_tr_sink = None
        self.emit("transcription_changed", sink)
        self._transcription_changed()
