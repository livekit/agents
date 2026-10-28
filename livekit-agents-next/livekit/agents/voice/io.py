"""The io legs a developer implements (an avatar's sink, a custom input), and the session's
``input``/``output`` slots that hand them to the core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import rtc

from .. import _core, llm, stt
from ..types import FlushSentinel, TimedString as TimedString

if TYPE_CHECKING:
    from .agent import ModelSettings
    from .agent_session import VoiceSession

STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame], "ModelSettings"],
    AsyncIterable[stt.SpeechEvent | str]
    | None
    | Awaitable[AsyncIterable[stt.SpeechEvent | str] | None],
]
LLMNode = Callable[
    [llm.ChatContext, list[llm.Tool], "ModelSettings"],
    AsyncIterable[llm.ChatChunk | str | FlushSentinel]
    | str
    | llm.ChatChunk
    | None
    | Awaitable[AsyncIterable[llm.ChatChunk | str | FlushSentinel] | str | llm.ChatChunk | None],
]
TTSNode = Callable[
    [AsyncIterable[str], "ModelSettings"],
    AsyncIterable[rtc.AudioFrame] | None | Awaitable[AsyncIterable[rtc.AudioFrame] | None],
]


class AudioInput:
    """A source of audio frames: iterate ``source``, or override ``__anext__``."""

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
        if self.source is None:
            raise NotImplementedError
        return await self.source.__anext__()

    def on_attached(self) -> None:
        if self.source:
            self.source.on_attached()

    def on_detached(self) -> None:
        if self.source:
            self.source.on_detached()


class VideoInput:
    """A source of video frames: iterate ``source``, or override ``__anext__``."""

    def __init__(self, *, label: str, source: VideoInput | None = None) -> None:
        self.__label = label
        self.__source = source

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame]:
        return self

    @property
    def label(self) -> str:
        return self.__label

    @property
    def source(self) -> VideoInput | None:
        return self.__source

    async def __anext__(self) -> rtc.VideoFrame:
        if self.source is None:
            raise NotImplementedError
        return await self.source.__anext__()

    def on_attached(self) -> None:
        if self.source:
            self.source.on_attached()

    def on_detached(self) -> None:
        if self.source:
            self.source.on_detached()


@dataclass
class PlaybackFinishedEvent:
    playback_position: float
    """How much of the audio was played back"""
    interrupted: bool
    """Interrupted is True if playback was interrupted (clear_buffer() was called)"""
    synchronized_transcript: str | None = None
    """Transcript synced with playback; may be partial if the audio was interrupted"""


@dataclass
class PlaybackStartedEvent:
    created_at: float
    """The timestamp (time.time()) when the playback started"""


@dataclass
class PlaybackProgressedEvent:
    """A stretch of the current segment that has played; never revised."""

    started_at: float
    offset: float
    duration: float


@dataclass
class AudioOutputCapabilities:
    pause: bool


_AudioEventT = TypeVar("_AudioEventT", default=Never, contravariant=True)


class AudioOutput(
    ABC,
    rtc.EventEmitter[
        Literal["playback_finished", "playback_started", "playback_progressed"] | _AudioEventT
    ],
    Generic[_AudioEventT],
):
    """An audio sink. Implement ``capture_frame``, ``flush`` and ``clear_buffer``, and report
    playback with ``on_playback_started`` / ``on_playback_finished``: the core paces the
    transcript and decides interruptions from those reports."""

    def __init__(
        self,
        *,
        label: str,
        capabilities: AudioOutputCapabilities,
        next_in_chain: AudioOutput | None = None,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__()
        self.__label = label
        self._capabilities = capabilities
        self._sample_rate = sample_rate
        self._next_in_chain = next_in_chain
        # the core's segment count and playout bookkeeping for this sink
        self._tracker = _core.PlaybackTracker()
        if next_in_chain is not None:
            next_in_chain.on("playback_started", lambda ev: self.on_playback_started(**vars(ev)))
            next_in_chain.on(
                "playback_progressed", lambda ev: self.on_playback_progressed(**vars(ev))
            )
            next_in_chain.on("playback_finished", lambda ev: self.on_playback_finished(**vars(ev)))

    @property
    def label(self) -> str:
        return self.__label

    @property
    def next_in_chain(self) -> AudioOutput | None:
        return self._next_in_chain

    @property
    def sample_rate(self) -> int | None:
        """The sample rate the sink requires; ``None`` takes any."""
        return self._sample_rate

    @property
    def can_pause(self) -> bool:
        return self._capabilities.pause and (not self.next_in_chain or self.next_in_chain.can_pause)

    def on_playback_started(self, *, created_at: float) -> None:
        self._tracker.playback_started(created_at)
        self.emit("playback_started", PlaybackStartedEvent(created_at=created_at))

    def on_playback_progressed(self, *, started_at: float, offset: float, duration: float) -> None:
        self._tracker.playback_progressed(started_at, offset, duration)
        self.emit(
            "playback_progressed",
            PlaybackProgressedEvent(started_at=started_at, offset=offset, duration=duration),
        )

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        """Call when a segment (ended by ``flush()`` or ``clear_buffer()``) has played out."""
        if self._tracker.playback_finished(playback_position, interrupted, synchronized_transcript):
            self.emit(
                "playback_finished",
                PlaybackFinishedEvent(playback_position, interrupted, synchronized_transcript),
            )

    async def wait_for_playout(self) -> PlaybackFinishedEvent:
        """Wait for the segments captured so far to finish playing out; returns the last one's
        event."""
        return PlaybackFinishedEvent(*await self._tracker.wait_for_playout())

    @abstractmethod
    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture an audio frame for playback; frames can be pushed faster than real time.
        Overrides call ``super()``: the first frame after a flush opens a segment."""
        self._tracker.note_capture()

    @abstractmethod
    def flush(self) -> None:
        """Mark the current segment complete. Overrides call ``super()``."""
        self._tracker.note_flush()

    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the buffer, stopping playback immediately."""

    def on_attached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_detached()

    def pause(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.pause()

    def resume(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.resume()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r}, next={self.next_in_chain!r})"


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
        """Capture a text segment (a ``TimedString`` when word timings are known)."""

    @abstractmethod
    def flush(self) -> None:
        """Mark the current text segment complete."""

    def on_attached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.on_detached()


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
    async def capture_frame(self, frame: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class AgentInput:
    """``session.input``: the legs the core reads. Before ``start()`` they are handed over
    with the session; after, each change is one command."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._audio: AudioInput | None = None
        self._video: VideoInput | None = None
        self._audio_enabled = True

    @property
    def audio(self) -> AudioInput | _core.RoomAudioInput | None:
        # the room's microphone when the host gave none, as ``RoomIO`` sets it
        if self._audio is None and self._session._core is not None:
            return self._session._core.audio_input
        return self._audio

    @audio.setter
    def audio(self, source: AudioInput | _core.RoomAudioInput | None) -> None:
        self._audio = source
        if self._session._core is not None:
            self._session._core.set_audio_input(source)

    @property
    def video(self) -> VideoInput | _core.RoomVideoInput | None:
        # the room's camera (``RoomOptions(video_input=True)``) when the host gave none, as
        # ``RoomIO`` sets it
        if self._video is None and self._session._core is not None:
            return self._session._core.video_input
        return self._video

    @video.setter
    def video(self, source: VideoInput | None) -> None:
        self._video = source
        if self._session._core is not None:
            self._session._core.set_video_input(source)

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = enabled
        self._session._running.set_io_enabled("audio_input", enabled)


class AgentOutput:
    """``session.output``: the sinks the core plays into."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._audio: AudioOutput | _core.DataStreamAudioOutput | None = None
        self._transcription: TextOutput | None = None
        self._audio_enabled = True
        self._transcription_enabled = True

    @property
    def audio(self) -> AudioOutput | _core.DataStreamAudioOutput | None:
        return self._audio

    @audio.setter
    def audio(self, sink: AudioOutput | _core.DataStreamAudioOutput | None) -> None:
        self._audio = sink
        if self._session._core is not None:
            self._session._core.set_audio_output(sink)

    def replace_audio_tail(self, sink: AudioOutput | _core.DataStreamAudioOutput) -> None:
        """Switch the sink at the bottom of the chain, keeping the wrappers above it: the core
        owns those (transcript synchroniser, recorder), so this is ``audio = sink``."""
        self.audio = sink

    @property
    def transcription(self) -> TextOutput | None:
        return self._transcription

    @transcription.setter
    def transcription(self, sink: TextOutput | None) -> None:
        self._transcription = sink
        if self._session._core is not None:
            self._session._core.set_transcription_output(sink)

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def transcription_enabled(self) -> bool:
        return self._transcription_enabled

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = enabled
        self._session._running.set_io_enabled("audio_output", enabled)

    def set_transcription_enabled(self, enabled: bool) -> None:
        self._transcription_enabled = enabled
        self._session._running.set_io_enabled("transcription", enabled)
