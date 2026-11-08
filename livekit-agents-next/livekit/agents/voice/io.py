from __future__ import annotations

import asyncio
import contextlib
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, Never, TypeAlias

from typing_extensions import TypeVar

from livekit import rtc

from .. import _core
from .._generated import (
    facades_voice_io as _facades,
    public_voice_io as _public,
    session_options as _session_options,
    speech_defaults,
)
from .._generated.io_events import PlaybackSegment as PlaybackSegment
from .._generated.public_voice_io import *  # noqa: F403
from ..log import logger
from ..types import TranscriptChunk as TranscriptChunk

if TYPE_CHECKING:
    from ..inference import TurnDetector
    from .agent_session import VoiceSession


_FrameT = TypeVar("_FrameT")
_SourceT = TypeVar("_SourceT", bound="_Input[Any, Any]")


class _Input(Generic[_FrameT, _SourceT]):
    def __init__(self, *, label: str, source: _SourceT | None = None) -> None:
        self.__label = label
        self.__source = source

    def __aiter__(self) -> AsyncIterator[_FrameT]:
        return self

    @property
    def label(self) -> str:
        return self.__label

    @property
    def source(self) -> _SourceT | None:
        return self.__source

    async def __anext__(self) -> _FrameT:
        if self.source is None:
            raise NotImplementedError
        return await self.source.__anext__()  # type: ignore[no-any-return]

    def on_attached(self) -> None:
        if self.source:
            self.source.on_attached()

    def on_detached(self) -> None:
        if self.source:
            self.source.on_detached()


class AudioInput(_Input[rtc.AudioFrame, "AudioInput"]):
    """A source of audio frames: iterate ``source``, or override ``__anext__``."""


class VideoInput(_Input[rtc.VideoFrameEvent, "VideoInput"]):
    """A source of ``rtc.VideoFrameEvent``: iterate ``source``, or override ``__anext__``."""


_AudioEventT = TypeVar("_AudioEventT", default=Never, contravariant=True)


class AudioOutput(
    ABC,
    rtc.EventEmitter[
        Literal["playback_finished", "playback_started", "playback_progressed"] | _AudioEventT
    ],
    Generic[_AudioEventT],
):
    """An audio sink. Implement ``capture_frame``, ``flush`` and ``clear_buffer``, and report
    playback with ``on_playback_started`` / ``on_playback_finished``: transcript pacing and
    interruptions are driven by those reports."""

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

    def begin_segment(self, segment: PlaybackSegment) -> None:
        """Admit the segment key before passing audio to a transport or a wrapper."""
        if not _core._core_playback_begin(self._tracker, segment):
            raise RuntimeError("the output is still capturing another segment")
        if self.next_in_chain is not None:
            self.next_in_chain.begin_segment(segment)

    @property
    def playback_segment(self) -> PlaybackSegment | None:
        """Latch this key when accepting audio and retain it with delayed playback reports."""
        return _core._core_playback_current_segment(self._tracker)

    def on_playback_started(self, *, segment: PlaybackSegment, created_at: float) -> None:
        if _core._core_playback_started(self._tracker, segment, created_at):
            self.emit(
                "playback_started", PlaybackStartedEvent(created_at=created_at, segment=segment)
            )

    def on_playback_progressed(
        self, *, segment: PlaybackSegment, started_at: float, offset: float, duration: float
    ) -> None:
        if _core._core_playback_progressed(self._tracker, segment, started_at, offset, duration):
            self.emit(
                "playback_progressed",
                PlaybackProgressedEvent(
                    started_at=started_at, offset=offset, duration=duration, segment=segment
                ),
            )

    def on_playback_finished(
        self,
        *,
        segment: PlaybackSegment,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        """Call when a segment (ended by ``flush()`` or ``clear_buffer()``) has played out."""
        if _core._core_playback_finished(
            self._tracker, segment, playback_position, interrupted, synchronized_transcript
        ):
            self.emit(
                "playback_finished",
                PlaybackFinishedEvent(
                    playback_position, interrupted, synchronized_transcript, segment
                ),
            )

    async def wait_for_playout(self) -> PlaybackFinishedEvent:
        """Wait for the segments captured so far to play out; returns the last one's event."""
        return await self._tracker.wait_for_playout()

    @abstractmethod
    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture an audio frame for playback; frames can be pushed faster than real time.
        Overrides call ``super()``: the first frame after a flush opens a segment."""
        _core._core_playback_note_capture(self._tracker)

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
        """Capture a plain text segment."""

    async def capture_transcript(self, chunk: TranscriptChunk) -> None:
        """Capture a transcript chunk; override to consume its alignment and speaker metadata.

        Plain text outputs receive the chunk's text by default.
        """
        await self.capture_text(chunk.text)

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
    """The session's inputs, exposed as ``session.input``."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._audio: AudioInput | _core.RoomAudioInput | None = None
        self._video: VideoInput | None = None
        self._audio_enabled = True

    @property
    def audio(self) -> AudioInput | _core.RoomAudioInput | None:
        # defaults to the room's microphone
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
        # defaults to the room's camera with ``RoomOptions(video_input=True)``
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
    """The session's outputs, exposed as ``session.output``."""

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
        """Replace the bottom audio sink; same as ``audio = sink``, the session keeps its
        wrappers (transcript sync, recorder) above it."""
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


class SpeechHandle(_facades.SpeechHandle):
    """One speech of the session. Awaiting it waits for playout and never raises; read
    ``exception()`` for a failure. The same object is returned wherever the speech appears
    (``speech_created``, a tool's ``RunContext``).

    ``done()`` turns true and ``wait_for_playout()`` returns first; done callbacks run after,
    once the callbacks of every earlier session event have run."""

    SPEECH_PRIORITY_LOW = speech_defaults.SPEECH_PRIORITY_LOW
    """Priority for messages played after all other messages in the queue."""
    SPEECH_PRIORITY_NORMAL = speech_defaults.SPEECH_PRIORITY_NORMAL
    """Default priority of every speech."""
    SPEECH_PRIORITY_HIGH = speech_defaults.SPEECH_PRIORITY_HIGH
    """Priority for important messages that should be played before others."""

    def __init__(self, core: _core.SpeechHandle) -> None:
        self._native = core
        # created lazily, once a done callback is added
        self._reported: asyncio.Future[None] | None = None
        core.host = weakref.ref(self)

    @staticmethod
    def _of(core: _core.SpeechHandle) -> SpeechHandle:
        ref = core.host
        handle = ref() if ref is not None else None
        return handle or SpeechHandle(core)

    async def wait_for_playout(self) -> None:
        from ._session_callbacks import _origin

        await self._native.wait_for_playout(origin=_origin())

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        """Run ``callback(handle)`` on the event loop once the speech has ended, after the
        callbacks of earlier session events; soon, if it already has. Callbacks run in the order
        added; one that raises is logged and the rest still run."""
        if self._reported is None:
            self._reported = asyncio.ensure_future(self._native.wait_reported())
        self._reported.add_done_callback(_DoneCallback(self, callback))

    def remove_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        """Forget every registration of ``callback`` that has not run yet."""
        if self._reported is not None:
            self._reported.remove_done_callback(_DoneCallback(self, callback))

    async def wait_if_not_interrupted(self, aw: Sequence[Awaitable[object]]) -> None:
        """Wait for ``aw``, unless the speech is interrupted first; ``aw`` is never cancelled."""
        gathered = asyncio.gather(*[asyncio.shield(fut) for fut in aw], return_exceptions=True)
        interrupted = asyncio.ensure_future(self._native.wait_for_interrupted())
        try:
            await asyncio.wait({gathered, interrupted}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for fut in (gathered, interrupted):
                if not fut.done():
                    fut.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await fut

    def __repr__(self) -> str:
        return f"SpeechHandle(id={self.id!r}, done={self.done()}, interrupted={self.interrupted})"


class _DoneCallback:
    # equal to any registration of the same callback, so `remove_done_callback` finds it
    __slots__ = ("callback", "handle")

    def __init__(self, handle: SpeechHandle, callback: Callable[[SpeechHandle], None]) -> None:
        self.handle = handle
        self.callback = callback

    def __call__(self, _: asyncio.Future[None]) -> None:
        try:
            self.callback(self.handle)
        except Exception as e:
            logger.warning(f"error in done_callback: {self.callback}", exc_info=e)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DoneCallback) and other.callback == self.callback

    __hash__ = None  # type: ignore[assignment]


TurnDetectionMode: TypeAlias = "_session_options.TurnDetectionMode | TurnDetector"
"""The mode of turn detection to use.

- "stt": use the speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manage turn detection manually
- an ``inference.TurnDetector``: predict the end of the user's turn from their audio

If not provided, the best mode is chosen from the available models. If the model a mode needs
(VAD, STT or RealtimeModel) is missing, the default mode is used.
"""


__all__ = [
    "TurnDetectionMode",
]
__all__ += _public.__all__
