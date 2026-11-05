"""The io legs a developer implements (an avatar's sink, a custom input), and the session's
``input``/``output`` slots that hand them to the core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import rtc
from livekit.agents._generated.playback_events import (
    PlaybackFinishedEvent as PlaybackFinishedEvent,
    PlaybackProgressedEvent as PlaybackProgressedEvent,
    PlaybackStartedEvent as PlaybackStartedEvent,
)

from .. import _core, llm, stt
from .._generated.io_events import PlaybackSegment as PlaybackSegment
from .._generated.rtc_wire import AudioOutputCapabilities as AudioOutputCapabilities
from ..types import FlushSentinel, TranscriptChunk as TranscriptChunk

if TYPE_CHECKING:
    from . import ModelSettings
    from .agent_session import VoiceSession
import asyncio
import contextlib
import weakref
from collections.abc import Generator, Sequence
from typing import Protocol, runtime_checkable

from livekit.agents._generated.speech_details import InputDetails as InputDetails
from livekit.agents._generated.turn_events import TurnDetectionEvent as TurnDetectionEvent

from .._generated import speech_defaults
from .._generated.language import LanguageCode
from .._generated.turn import (
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions,
    TurnHandlingOptions,
    UserTurnLimitOptions,
)
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

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
        """Wait for the segments captured so far to finish playing out; returns the last one's
        event."""
        return PlaybackFinishedEvent(*await self._tracker.wait_for_playout())

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
    """``session.input``: the legs the core reads. Before ``start()`` they are handed over
    with the session; after, each change is one command."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session
        self._audio: AudioInput | _core.RoomAudioInput | None = None
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


DEFAULT_INPUT_DETAILS = InputDetails(modality=speech_defaults.DEFAULT_INPUT_MODALITY)

InterruptionSource = Literal["audio_activity", "user_turn", "programmatic"]
"""Why a speech was interrupted, as the reference's ``agent_turn`` trace names it."""


class SpeechHandle:
    """One speech of the session, over the core's handle: its state is the core's, and awaiting
    it waits for the core to finish playing it out. Awaiting never raises; read
    ``exception()`` for a failure. The core hands this object back wherever the speech is named
    (``speech_created``, a tool's ``RunContext``), for as long as it lives."""

    SPEECH_PRIORITY_LOW = speech_defaults.SPEECH_PRIORITY_LOW
    """Priority for messages that should be played after all other messages in the queue"""
    SPEECH_PRIORITY_NORMAL = speech_defaults.SPEECH_PRIORITY_NORMAL
    """Every speech generates by the VoiceAgent defaults to this priority."""
    SPEECH_PRIORITY_HIGH = speech_defaults.SPEECH_PRIORITY_HIGH
    """Priority for important messages that should be played before others."""

    def __init__(self, core: _core.SpeechHandle, *, host_done: bool = False) -> None:
        self._core = core
        self._done_callbacks: list[Callable[[SpeechHandle], None]] = []
        # The session's `speech_done` reached this handle. Never inferred from the core's
        # `done()`, which turns true before `speech_done` reaches the host loop.
        self._host_done = host_done
        core.host = weakref.ref(self)

    @staticmethod
    def _of(core: _core.SpeechHandle, *, reported: bool = True) -> SpeechHandle:
        """The handle over ``core``: the one already made, else a new one. The session holds a
        speech's handle from ``speech_created`` to ``speech_done``, so a new one for an ended
        speech starts done; ``speech_created`` itself passes ``reported=False``."""
        ref = core.host
        handle = ref() if ref is not None else None
        return handle or SpeechHandle(core, host_done=reported and core.done())

    @property
    def id(self) -> str:
        return self._core.id

    @property
    def num_steps(self) -> int:
        return self._core.num_steps

    @property
    def input_details(self) -> InputDetails:
        """How the input this speech answers arrived: the core runs every speech as audio."""
        return DEFAULT_INPUT_DETAILS

    @property
    def scheduled(self) -> bool:
        return self._core.scheduled

    @property
    def interrupted(self) -> bool:
        return self._core.interrupted

    @property
    def allow_interruptions(self) -> bool:
        return self._core.allow_interruptions

    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None:
        self._core.allow_interruptions = value

    @property
    def chat_items(self) -> list[llm.ChatItem]:

        return list(self._core.chat_items)

    def done(self) -> bool:
        return self._core.done()

    def exception(self) -> BaseException | None:
        """The error the speech failed with, or ``None``.

        Raises:
            asyncio.InvalidStateError: If the speech is not done yet.
        """
        if not self.done():
            raise asyncio.InvalidStateError("SpeechHandle is not done yet")
        return self._core.exception()

    def interrupt(
        self, *, force: bool = False, source: InterruptionSource = "programmatic"
    ) -> SpeechHandle:
        """Interrupt the speech. ``source`` is accepted for the reference's signature: the trace
        records the cause the core observed."""
        self._core.interrupt(force=force)
        return self

    async def wait_for_playout(self) -> None:
        from .agent_session import _origin

        await self._core.wait_for_playout(origin=_origin())

    async def _wait_for_generation(self, step_idx: int = -1) -> None:
        await self._core.wait_for_generation(None if step_idx < 0 else step_idx)

    def __await__(self) -> Generator[None, None, SpeechHandle]:
        async def _await() -> SpeechHandle:
            await self.wait_for_playout()
            return self

        return _await().__await__()

    def add_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        """``callback(handle)`` once the session reported the speech done (``speech_done``); soon,
        if it already did. Every callback runs on that one event, in the order added."""
        if self._host_done:
            asyncio.get_running_loop().call_soon(callback, self)
            return
        self._done_callbacks.append(callback)

    def remove_done_callback(self, callback: Callable[[SpeechHandle], None]) -> None:
        with contextlib.suppress(ValueError):
            self._done_callbacks.remove(callback)

    def _mark_done(self) -> None:
        """The core reported the speech done (``speech_done``): its callbacks run, in order."""
        self._host_done = True
        callbacks, self._done_callbacks = self._done_callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"error in done_callback: {callback}", exc_info=e)

    async def wait_if_not_interrupted(self, aw: Sequence[Awaitable[object]]) -> None:
        """Wait for ``aw``, unless the speech is interrupted first; ``aw`` is never cancelled."""
        gathered = asyncio.gather(*[asyncio.shield(fut) for fut in aw], return_exceptions=True)
        interrupted = asyncio.ensure_future(self._core.wait_for_interrupted())
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


@runtime_checkable
class _StreamingTurnDetectorStream(Protocol):
    """I/O stream for the streaming turn detector."""

    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def is_fallback(self) -> bool: ...
    @property
    def prediction_timeout(self) -> float: ...

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def supports_language(self, language: LanguageCode | None) -> bool: ...

    def predict(self) -> asyncio.Future[TurnDetectionEvent]: ...
    def cancel_inference(self, *, timed_out: bool = False) -> None: ...
    def flush(self, reason: str | None = None) -> None: ...
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...
    def end_input(self) -> None: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class _StreamingTurnDetector(Protocol):
    """Turn detector that processes streaming data."""

    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _StreamingTurnDetectorStream: ...


TurnDetectionMode = Literal["stt", "vad", "realtime_llm", "manual"] | _StreamingTurnDetector
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- a turn detector (``inference.TurnDetector`` by default): the default mode, the end of the
  user's turn predicted on their audio

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""


__all__ = [
    "EndpointingOptions",
    "InterruptionOptions",
    "PreemptiveGenerationOptions",
    "TurnDetectionMode",
    "TurnHandlingOptions",
    "UserTurnLimitOptions",
]
