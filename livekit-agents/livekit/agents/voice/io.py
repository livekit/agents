from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from livekit import rtc

from .. import llm, stt
from ..log import logger
from ..types import FlushSentinel, TimedString as TimedString
from .agent import ModelSettings

# TODO(theomonnom): can those types be simplified?
STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame], ModelSettings],
    AsyncIterable[stt.SpeechEvent | str]
    | None
    | Awaitable[AsyncIterable[stt.SpeechEvent | str] | None],
]
LLMNode = Callable[
    [
        llm.ChatContext,
        list[llm.Tool],
        ModelSettings,
    ],
    AsyncIterable[llm.ChatChunk | str | FlushSentinel]
    | str
    | llm.ChatChunk
    | None
    | Awaitable[AsyncIterable[llm.ChatChunk | str | FlushSentinel] | str | llm.ChatChunk | None],
]
TTSNode = Callable[
    [AsyncIterable[str], ModelSettings],
    AsyncIterable[rtc.AudioFrame] | None | Awaitable[AsyncIterable[rtc.AudioFrame] | None],
]


class AudioInput:
    def __init__(self, *, label: str, source: AudioInput | None = None) -> None:
        self.__label = label
        self.__source = source

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    @property
    def label(self) -> str:
        return self.__label

    @property
    def source(self) -> AudioInput | None:
        return self.__source

    async def __anext__(self) -> rtc.AudioFrame:
        if self.source:
            return await self.source.__anext__()

        raise NotImplementedError

    def on_attached(self) -> None:
        if self.source:
            self.source.on_attached()

    def on_detached(self) -> None:
        if self.source:
            self.source.on_detached()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r}, source={self.source!r})"


class VideoInput:
    def __init__(self, *, label: str, source: VideoInput | None = None) -> None:
        self.__source = source
        self.__label = label

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame]:
        return self

    @property
    def label(self) -> str:
        return self.__label

    @property
    def source(self) -> VideoInput | None:
        return self.__source

    async def __anext__(self) -> rtc.VideoFrame:
        if self.source:
            return await self.source.__anext__()

        raise NotImplementedError

    def on_attached(self) -> None:
        if self.source:
            self.source.on_attached()

    def on_detached(self) -> None:
        if self.source:
            self.source.on_detached()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r}, source={self.source!r})"


@dataclass
class PlaybackFinishedEvent:
    playback_position: float
    """How much of the audio was played back"""
    interrupted: bool
    """Interrupted is True if playback was interrupted (clear_buffer() was called)"""
    synchronized_transcript: str | None = None
    """Transcript synced with playback; may be partial if the audio was interrupted
    When None, the transcript is not synchronized with the playback"""


@dataclass
class PlaybackStartedEvent:
    created_at: float
    """The timestamp (time.time())when the playback started"""


@dataclass
class AudioOutputCapabilities:
    pause: bool


class AudioOutput(ABC, rtc.EventEmitter[Literal["playback_finished", "playback_started"]]):
    def __init__(
        self,
        *,
        label: str,
        capabilities: AudioOutputCapabilities,
        next_in_chain: AudioOutput | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """
        Args:
            sample_rate: The sample rate required by the audio sink, if None, any sample rate is accepted
        """  # noqa: E501
        super().__init__()
        self._sample_rate = sample_rate
        self.__label = label
        self.__capturing = False
        self.__playback_finished_event = asyncio.Event()
        self._capabilities = capabilities

        self.__playback_segments_count = 0
        self.__playback_finished_count = 0
        self.__last_playback_ev: PlaybackFinishedEvent = PlaybackFinishedEvent(
            playback_position=0, interrupted=False
        )

        # auto-wrap a bare leaf with a _AudioSinkProxy so the leaf can be
        # hot-swapped later without disturbing wrappers above. wrappers that
        # cache next_in_chain (e.g. _SyncedAudioOutput) cache the proxy, so
        # their references stay valid across swaps
        if (
            next_in_chain is not None
            and next_in_chain.next_in_chain is None
            and not isinstance(next_in_chain, _AudioSinkProxy)
        ):
            next_in_chain = _AudioSinkProxy(next_in_chain)

        self._next_in_chain: AudioOutput | None = next_in_chain
        if next_in_chain is not None:
            next_in_chain.on("playback_finished", self._forward_next_playback_finished)
            next_in_chain.on("playback_started", self._forward_next_playback_started)

    def _forward_next_playback_finished(self, ev: PlaybackFinishedEvent) -> None:
        self.on_playback_finished(
            interrupted=ev.interrupted,
            playback_position=ev.playback_position,
            synchronized_transcript=ev.synchronized_transcript,
        )

    def _forward_next_playback_started(self, ev: PlaybackStartedEvent) -> None:
        self.on_playback_started(created_at=ev.created_at)

    @property
    def label(self) -> str:
        return self.__label

    @property
    def next_in_chain(self) -> AudioOutput | None:
        return self._next_in_chain

    def on_playback_started(self, *, created_at: float) -> None:
        self.emit("playback_started", PlaybackStartedEvent(created_at=created_at))

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
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

        ev = PlaybackFinishedEvent(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
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
        target = self.__playback_segments_count

        while self.__playback_finished_count < target:
            await self.__playback_finished_event.wait()
            self.__playback_finished_event.clear()

        return self.__last_playback_ev

    def _reset_playback_count(self) -> None:
        self.__playback_segments_count = 0
        self.__playback_finished_count = 0

    @property
    def sample_rate(self) -> int | None:
        """The sample rate required by the audio sink, if None, any sample rate is accepted"""
        return self._sample_rate

    @property
    def can_pause(self) -> bool:
        return self._capabilities.pause and (not self.next_in_chain or self.next_in_chain.can_pause)

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

    def on_attached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_detached()

    def pause(self) -> None:
        """Pause the audio playback"""
        if self.next_in_chain:
            self.next_in_chain.pause()

    def resume(self) -> None:
        """Resume the audio playback"""
        if self.next_in_chain:
            self.next_in_chain.resume()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r}, next={self.next_in_chain!r})"


class _AudioSinkProxy(AudioOutput):
    """Stable swap point at the bottom of an audio wrapper chain.

    Wrappers above hold a reference to the proxy; the actual sink lives in
    ``next_in_chain`` and can be replaced via :meth:`set_next_in_chain` without
    disturbing them. When detached (``next_in_chain`` is None), the proxy acts
    as a no-op sink that still cooperates with the playback-finished protocol
    so upstream wrappers don't hang.

    Only the proxy has a mutable ``next_in_chain`` — regular ``AudioOutput``
    subclasses store theirs immutably at construction.
    """

    def __init__(self, next_in_chain: AudioOutput | None = None) -> None:
        super().__init__(
            label="AudioSinkProxy",
            capabilities=AudioOutputCapabilities(pause=True),
            next_in_chain=None,
        )
        # whether the wrapper above us has attached the proxy; set_next_in_chain
        # uses this to decide if a new/old downstream should be notified
        self._attached = False
        if next_in_chain is not None:
            self.set_next_in_chain(next_in_chain)

    def on_attached(self) -> None:
        self._attached = True
        super().on_attached()

    def on_detached(self) -> None:
        self._attached = False
        super().on_detached()

    def set_next_in_chain(self, new: AudioOutput | None) -> None:
        """Replace the downstream sink, transferring playback listeners
        and on_attached/on_detached state.

        The new sink must be a leaf (no ``next_in_chain``) — the proxy is
        the bottom of a wrapper chain and cannot itself hold wrappers.
        """
        assert new is None or new.next_in_chain is None, (
            "_AudioSinkProxy.set_next_in_chain expects a leaf sink, not a wrapper chain"
        )
        if new is self._next_in_chain:
            return

        old = self._next_in_chain
        if old is not None:
            old.off("playback_finished", self._forward_next_playback_finished)
            old.off("playback_started", self._forward_next_playback_started)
            if self._attached:
                old.on_detached()

        self._next_in_chain = new

        if new is not None:
            new.on("playback_finished", self._forward_next_playback_finished)
            new.on("playback_started", self._forward_next_playback_started)
            if self._attached:
                new.on_attached()

    @property
    def sample_rate(self) -> int | None:
        return self.next_in_chain.sample_rate if self.next_in_chain else None

    @property
    def can_pause(self) -> bool:
        return not self.next_in_chain or self.next_in_chain.can_pause

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        if self.next_in_chain:
            self.next_in_chain.flush()
        else:
            # no real sink; synthesize a playback_finished
            self.on_playback_finished(playback_position=0.0, interrupted=True)

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()


class TextOutput(ABC):
    def __init__(self, *, label: str, next_in_chain: TextOutput | None) -> None:
        self.__label = label
        self.__next_in_chain = next_in_chain

    @property
    def label(self) -> str:
        return self.__label

    @property
    def next_in_chain(self) -> TextOutput | None:
        return self.__next_in_chain

    @abstractmethod
    async def capture_text(self, text: str) -> None:
        """Capture a text segment (Used by the output of LLM nodes)"""

    @abstractmethod
    def flush(self) -> None:
        """Mark the current text segment as complete (e.g LLM generation is complete)"""

    def on_attached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_detached()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r}, next={self.next_in_chain!r})"


# TODO(theomonnom): Add documentation to VideoSink
class VideoOutput(ABC):
    def __init__(self, *, label: str, next_in_chain: VideoOutput | None) -> None:
        self.__label = label
        self.__next_in_chain = next_in_chain

    @property
    def label(self) -> str:
        return self.__label

    @property
    def next_in_chain(self) -> VideoOutput | None:
        return self.__next_in_chain

    @abstractmethod
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    def on_attached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_detached()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r}, next={self.next_in_chain!r})"


class AgentInput:
    def __init__(
        self,
        video_changed: Callable[[], None],
        audio_changed: Callable[[], None],
        audio_enabled_cb: Callable[[bool], None] | None = None,
    ) -> None:
        self._video_stream: VideoInput | None = None
        self._audio_stream: AudioInput | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._audio_enabled_cb = audio_enabled_cb

        # enabled by default
        self._audio_enabled = True
        self._video_enabled = True

    def set_audio_enabled(self, enable: bool) -> None:
        if enable and not self._audio_stream:
            logger.warning("Cannot enable audio input when it's not set")

        if enable == self._audio_enabled:
            return

        self._audio_enabled = enable

        if self._audio_enabled_cb is not None:
            self._audio_enabled_cb(enable)

        if not self._audio_stream:
            return

        if enable:
            self._audio_stream.on_attached()
        else:
            self._audio_stream.on_detached()

    def set_video_enabled(self, enable: bool) -> None:
        if enable and not self._video_stream:
            logger.warning("Cannot enable video input when it's not set")

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
        if stream is self._video_stream:
            return

        if self._video_stream:
            self._video_stream.on_detached()

        self._video_stream = stream
        self._video_changed()

        if self._video_stream:
            if self._video_enabled:
                self._video_stream.on_attached()
            else:
                self._video_stream.on_detached()

    @property
    def audio(self) -> AudioInput | None:
        return self._audio_stream

    @audio.setter
    def audio(self, stream: AudioInput | None) -> None:
        if stream is self._audio_stream:
            return

        if self._audio_stream:
            self._audio_stream.on_detached()

        self._audio_stream = stream
        self._audio_changed()

        if self._audio_stream:
            if self._audio_enabled:
                self._audio_stream.on_attached()
            else:
                self._audio_stream.on_detached()


class AgentOutput:
    def __init__(
        self,
        video_changed: Callable[[], None],
        audio_changed: Callable[[], None],
        transcription_changed: Callable[[], None],
    ) -> None:
        self._video_sink: VideoOutput | None = None
        self._audio_sink: AudioOutput | None = None
        self._transcription_sink: TextOutput | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._transcription_changed = transcription_changed

        self._audio_enabled = True
        self._video_enabled = True
        self._transcription_enabled = True

    def set_video_enabled(self, enabled: bool) -> None:
        if enabled and not self._video_sink:
            logger.warning("Cannot enable video output when it's not set")

        if enabled == self._video_enabled:
            return

        self._video_enabled = enabled

        if not self._video_sink:
            return

        if enabled:
            self._video_sink.on_attached()
        else:
            self._video_sink.on_detached()

    def set_audio_enabled(self, enabled: bool) -> None:
        if enabled and not self._audio_sink:
            logger.warning("Cannot enable audio output when it's not set")

        if enabled == self._audio_enabled:
            return

        self._audio_enabled = enabled

        if not self._audio_sink:
            return

        if enabled:
            self._audio_sink.on_attached()
        else:
            self._audio_sink.on_detached()

    def set_transcription_enabled(self, enabled: bool) -> None:
        if enabled and not self._transcription_sink:
            logger.warning("Cannot enable transcription output when it's not set")

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
        if sink is self._video_sink:
            return

        if self._video_sink:
            self._video_sink.on_detached()

        self._video_sink = sink
        self._video_changed()

        if self._video_sink:
            if self._video_enabled:
                self._video_sink.on_attached()
            else:
                self._video_sink.on_detached()

    @property
    def audio(self) -> AudioOutput | None:
        return self._audio_sink

    @audio.setter
    def audio(self, sink: AudioOutput | None) -> None:
        if sink is self._audio_sink:
            return

        if self._audio_sink:
            self._audio_sink.on_detached()

        self._audio_sink = sink
        self._audio_changed()

        if self._audio_sink:
            if self._audio_enabled:
                self._audio_sink.on_attached()
            else:
                self._audio_sink.on_detached()

    def set_audio_sink(self, sink: AudioOutput | None, *, preserve_wrappers: bool = False) -> None:
        """Set the audio sink at the bottom of the chain.

        When ``preserve_wrappers`` is True, walks the chain looking for a
        :class:`_AudioSinkProxy` and swaps its downstream — leaving wrappers
        like :class:`TranscriptSynchronizer` and :class:`RecorderAudioOutput`
        attached. Falls back to ``self.audio = sink`` if no proxy is present
        (no wrappers, or the chain hasn't been set up yet).

        With the default ``preserve_wrappers=False``, this is exactly
        equivalent to ``self.audio = sink``.
        """
        if preserve_wrappers:
            cur = self._audio_sink
            while cur is not None:
                if isinstance(cur, _AudioSinkProxy):
                    cur.set_next_in_chain(sink)
                    return
                cur = cur.next_in_chain
        self.audio = sink

    @property
    def transcription(self) -> TextOutput | None:
        return self._transcription_sink

    @transcription.setter
    def transcription(self, sink: TextOutput | None) -> None:
        if sink is self._transcription_sink:
            return

        if self._transcription_sink:
            self._transcription_sink.on_detached()

        self._transcription_sink = sink
        self._transcription_changed()

        if self._transcription_sink:
            if self._transcription_enabled:
                self._transcription_sink.on_attached()
            else:
                self._transcription_sink.on_detached()
