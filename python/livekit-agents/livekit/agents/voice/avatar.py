from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import rtc

from .. import _core, utils
from .._generated import public_voice_avatar as _public
from .._generated.avatar_queue import QueueEffects
from .._generated.io_events import AvatarPlaybackSegment
from .._generated.public_voice_avatar import *  # noqa: F403
from ..job import get_job_context
from ..types import NOT_GIVEN, NotGivenOr
from .io import AudioOutput, AudioOutputCapabilities

if TYPE_CHECKING:
    from .agent_session import VoiceSession

# Agent <-> avatar worker protocol over DataStream: one byte stream per segment on
# `lk.audio_stream`, `lk.clear_buffer` from the agent, `lk.playback_started` and
# `lk.playback_finished` back. Both ends and the runner live in the core.

__all__ = [
    "AvatarRunner",
    "AvatarSession",
    "VideoGenerator",
    "AudioReceiver",
    "QueueAudioOutput",
    "DataStreamAudioReceiver",
]
__all__ += _public.__all__


_ReceiverEventT = TypeVar("_ReceiverEventT", default=Never, contravariant=True)


class AudioReceiver(
    ABC, rtc.EventEmitter[Literal["clear_buffer"] | _ReceiverEventT], Generic[_ReceiverEventT]
):
    """Audio the avatar worker receives from the agent, fed by ``AvatarRunner`` to its generator.

    A subclass yields frames, then ``AudioSegmentEnd`` at each segment's end. Emitting
    ``"clear_buffer"`` cuts the current segment. ``notify_playback_started`` and
    ``notify_playback_finished`` receive the segment's key.
    """

    async def start(self) -> None:
        pass

    @abstractmethod
    def notify_playback_finished(
        self, playback_position: float, interrupted: bool, *, segment: AvatarPlaybackSegment
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has finished"""

    @abstractmethod
    def notify_playback_started(
        self, *, segment: AvatarPlaybackSegment
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has started"""

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


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class AvatarSession(ABC, rtc.EventEmitter[Literal["metrics_collected"] | TEvent], Generic[TEvent]):
    """Base class for avatar plugin sessions.

    ``start()`` begins the avatar's lifecycle:

    - The avatar participant (``avatar_identity``) is awaited in the background until it joins
      and publishes its video; ``wait_for_join()`` awaits it.
    - Inside a job, the job's shutdown runs ``aclose()``.
    - ``metrics_collected`` emits ``AvatarMetrics`` once when the avatar joins, then for each
      assistant message with a ``playback_latency`` (first audio frame sent to playback start),
      until ``aclose()``.

    Subclasses call ``super().start()`` and ``super().aclose()``; calling ``start()`` again
    after ``aclose()`` begins a new lifecycle.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lifecycle: _core._core_AvatarLifecycle | None = None
        self._room: rtc.Room | None = None
        self._agent_session: VoiceSession | None = None
        self._audio_output: DataStreamAudioOutput | None = None

    @property
    @abstractmethod
    def avatar_identity(self) -> str:
        """The participant identifier of the avatar"""
        ...

    @property
    def provider(self) -> str:
        """The provider of the avatar"""
        return "unknown"

    async def start(self, agent_session: VoiceSession, room: rtc.Room) -> None:
        self._room = room
        self._agent_session = agent_session
        self._lifecycle = _core._core_AvatarLifecycle(
            agent_session._native,
            room,
            identity=self.avatar_identity,
            provider=self.provider,
            avatar=self,
            job=get_job_context(required=False),
        )

    def _attach_audio_output(
        self, agent_session: VoiceSession, output: DataStreamAudioOutput
    ) -> None:
        if self._audio_output is not None and self._audio_output is not output:
            raise RuntimeError("the avatar already owns an audio output")
        self._audio_output = output
        if self._lifecycle is not None:
            # past the join timeout, its audio fails at once instead of waiting for the avatar
            self._lifecycle.attach_output(output)
        agent_session.output.replace_audio_tail(output)

    async def wait_for_join(self, *, timeout: float | None = 30.0) -> None:
        """Wait until the avatar participant has joined the room and published its video track.

        Returns at once before ``start()`` or after ``aclose()``.

        Raises:
            TimeoutError: Not joined within ``timeout`` seconds (``None`` waits indefinitely);
                the background wait continues.
            asyncio.CancelledError: The avatar was closed before it joined.
        """
        if self._lifecycle is None:
            return
        await self._lifecycle.wait_for_join(timeout)

    async def aclose(self) -> None:
        """Close the avatar's audio output and lifecycle; inside a job, also remove the avatar
        participant from the room. Idempotent."""
        try:
            output = self._audio_output
            if output is not None:
                await output.aclose()
                if self._audio_output is output:
                    self._audio_output = None
        finally:
            lifecycle = self._lifecycle
            if lifecycle is not None:
                # a concurrent close waits for the same one
                await lifecycle.aclose()
            if self._lifecycle is lifecycle:
                self._lifecycle = None
                self._agent_session = None
                self._room = None


class DataStreamAudioReceiver(AudioReceiver):
    """Receives the agent's audio over LiveKit DataStream.

    Subscribes to ``sender_identity`` if given, else to the first agent participant in the room.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        sender_identity: str | None = None,
        frame_size_ms: NotGivenOr[int] = NOT_GIVEN,
        rpc_max_retries: int = 3,
    ):
        super().__init__()
        self._native = _core.DataStreamAudioReceiver(
            room,
            sender_identity=sender_identity,
            frame_size_ms=frame_size_ms if utils.is_given(frame_size_ms) else 100,
            rpc_max_retries=rpc_max_retries,
            on_clear_buffer=self,
        )

    async def start(self) -> None:
        await self._native.start()

    def notify_playback_finished(
        self, playback_position: float, interrupted: bool, *, segment: AvatarPlaybackSegment
    ) -> None:
        self._native.notify_playback_finished(segment, playback_position, interrupted)

    def notify_playback_started(self, *, segment: AvatarPlaybackSegment) -> None:
        self._native.notify_playback_started(segment)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        if (item := (await self._native.next_audio()).frame) is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._native.aclose()


class QueueAudioOutput(
    AudioOutput[Literal["clear_buffer"]],
    AudioReceiver[Literal["playback_finished", "playback_started", "playback_progressed"]],
):
    """An in-process audio output that is also the avatar's ``AudioReceiver``."""

    def __init__(
        self, *, sample_rate: int | None = None, wait_playback_start: bool = False
    ) -> None:
        super().__init__(
            label="DebugQueueIO",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self._native = _core._core_QueueAudio(wait_playback_start)

    def _apply(self, effects: QueueEffects) -> None:
        if effects.flushed:
            self._tracker.note_flush()
        for segment in effects.started:
            self.on_playback_started(segment=segment, created_at=time.time())
        for finished in effects.finished:
            self.on_playback_finished(
                segment=finished.segment,
                playback_position=finished.playback_position,
                interrupted=finished.interrupted,
            )
        if effects.clear:
            self.emit("clear_buffer")
        if effects.reset:
            self._tracker.reset()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self._native.check_open()
        await super().capture_frame(frame)
        key = self.playback_segment
        assert key is not None
        self._apply(self._native.capture(frame, key))

    def flush(self) -> None:
        super().flush()
        self._native.flush()

    def clear_buffer(self) -> None:
        self._apply(self._native.clear())

    def notify_playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
        *,
        segment: AvatarPlaybackSegment | None = None,
    ) -> None:
        self._apply(self._native.playback_finished(playback_position, interrupted, segment))

    def notify_playback_started(self, *, segment: AvatarPlaybackSegment | None = None) -> None:
        self._apply(self._native.playback_started(segment))

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        if (item := (await self._native.next_audio()).frame) is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._apply(self._native.close())


class AvatarRunner:
    """Worker that generates synchronized avatar video from received audio."""

    def __init__(
        self,
        room: rtc.Room,
        *,
        audio_recv: AudioReceiver,
        video_gen: VideoGenerator,
        options: AvatarOptions,
        # queue size of the AV synchronizer
        _queue_size_ms: int = 100,
        # defer publishing the tracks until the first frame is pushed
        _lazy_publish: bool = True,
    ) -> None:
        self._audio_recv = audio_recv
        self._host_receiver: _core.HostAudioReceiver | None = None
        receiver: _core.DataStreamAudioReceiver | _core.HostAudioReceiver
        if isinstance(audio_recv, DataStreamAudioReceiver):
            receiver = audio_recv._native
        elif isinstance(audio_recv, QueueAudioOutput):
            # the queue stamps and reports its own clears
            receiver = _core.HostAudioReceiver(audio_recv, audio=audio_recv._native)
        else:
            receiver = self._host_receiver = _core.HostAudioReceiver(audio_recv)
        self._core = _core.AvatarRunner(
            _core._core_create_room_session(room),
            receiver=receiver,
            generator=video_gen,
            options=options,
            queue_size_ms=_queue_size_ms,
            lazy_publish=_lazy_publish,
        )

    @property
    def av_sync(self) -> _core.AVSynchronizer:
        return self._core.av_sync

    async def start(self) -> None:
        await self._audio_recv.start()
        if self._host_receiver is not None:
            self._audio_recv.on("clear_buffer", self._host_receiver.clear_buffer)
        await self._core.start()

    async def wait_for_complete(self) -> None:
        await self._core.wait_for_complete()

    async def aclose(self) -> None:
        try:
            await self._core.aclose()
        finally:
            if self._host_receiver is not None:
                self._audio_recv.off("clear_buffer", self._host_receiver.clear_buffer)
