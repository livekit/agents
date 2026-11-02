"""Avatars: the session and its video generator, the runner an avatar worker drives, the queue
audio output, and the agent and an avatar worker exchanging audio over LiveKit DataStream: one
byte stream per segment on ``lk.audio_stream``, ``lk.clear_buffer`` from the agent,
``lk.playback_started`` and ``lk.playback_finished`` back. Both DataStream ends run in the core
(``_core.DataStreamAudioOutput``, ``_core.DataStreamAudioReceiver``), as does the runner
(``_core.AvatarRunner``): RTC source capabilities, native pacing and queue state are shared
with the Rust runtime. Python adapts user callbacks and delivers SDK events."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import rtc
from livekit.api import TwirpError

from .. import _core, utils
from .._generated.avatar import AudioSegmentEnd as AudioSegmentEnd
from .._generated.avatar_options import AvatarOptions
from .._generated.avatar_queue import QueueEffects
from .._generated.io_events import AvatarPlaybackSegment
from ..job import get_job_context
from ..metrics import AvatarMetrics, Metadata
from ..types import NOT_GIVEN, NotGivenOr
from .events import ConversationItemAddedEvent
from .io import AudioOutput, AudioOutputCapabilities
from .room_io import Room

if TYPE_CHECKING:
    from .agent_session import VoiceSession

logger = logging.getLogger(__name__)

__all__ = [
    "AvatarRunner",
    "AvatarOptions",
    "AvatarSession",
    "VideoGenerator",
    "AudioReceiver",
    "AudioSegmentEnd",
    "QueueAudioOutput",
    "DataStreamAudioReceiver",
    "DataStreamAudioOutput",
]


_ReceiverEventT = TypeVar("_ReceiverEventT", default=Never, contravariant=True)


class AudioReceiver(
    ABC, rtc.EventEmitter[Literal["clear_buffer"] | _ReceiverEventT], Generic[_ReceiverEventT]
):
    _core_audio_epoch = 0

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

    # what `_core.AvatarRunner` calls (`AvatarAudioReceiver`)

    def _core_clear_buffer(self) -> None:
        self.emit("clear_buffer")

    async def _core_open_audio(self) -> None:
        self._core_audio = self.__aiter__()

    async def _core_next_audio(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment | None, rtc.AudioFrame | AudioSegmentEnd | None]:
        raise RuntimeError("a custom audio receiver must supply a keyed audio receipt")

    async def _core_notify_playback_started(self, segment: AvatarPlaybackSegment) -> None:
        await _maybe_await(self.notify_playback_started(segment=segment))

    async def _core_notify_playback_finished(
        self, segment: AvatarPlaybackSegment, playback_position: float, interrupted: bool
    ) -> None:
        await _maybe_await(
            self.notify_playback_finished(playback_position, interrupted, segment=segment)
        )

    async def _core_aclose(self) -> None:
        await self.aclose()


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

    # what `_core.AvatarRunner` calls (`AvatarVideoGenerator`)

    async def _core_push_audio(self, frame: rtc.AudioFrame | None) -> None:
        await self.push_audio(AudioSegmentEnd() if frame is None else frame)

    async def _core_clear_buffer(self) -> None:
        await _maybe_await(self.clear_buffer())

    async def _core_open_frames(self) -> None:
        self._core_frames = self.__aiter__()

    async def _core_next_frame(self) -> rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd | None:
        return await anext(self._core_frames, None)


async def _maybe_await(result: None | Coroutine[None, None, None]) -> None:
    # a receiver or generator may answer synchronously or with a coroutine
    if asyncio.iscoroutine(result):
        await result


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class AvatarSession(ABC, rtc.EventEmitter[Literal["metrics_collected"] | TEvent], Generic[TEvent]):
    """Base class for avatar plugin sessions."""

    def __init__(self) -> None:
        super().__init__()
        self._wait_avatar_join_task: asyncio.Task[None] | None = None
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
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            job_ctx.add_shutdown_callback(self.aclose)
        else:
            logger.debug(
                "AvatarSession started outside a job context; call aclose() manually to "
                "release resources when the job shuts down"
            )

        self._room = room
        self._agent_session = agent_session
        self._agent_session.on("conversation_item_added", self._on_conversation_item_added)
        self._wait_avatar_join_task = asyncio.create_task(self._wait_avatar_join(room))

    def _attach_audio_output(
        self, agent_session: VoiceSession, output: DataStreamAudioOutput
    ) -> None:
        if self._audio_output is not None and self._audio_output is not output:
            raise RuntimeError("the avatar already owns an audio output")
        self._audio_output = output
        agent_session.output.replace_audio_tail(output)

    async def wait_for_join(self, *, timeout: float | None = 30.0) -> None:
        """Wait until the avatar participant has joined the room and
        published its video track.

        Raises ``asyncio.TimeoutError`` if it doesn't arrive within
        ``timeout`` seconds. Pass ``timeout=None`` to wait indefinitely.
        """
        if self._wait_avatar_join_task is None:
            # not started
            return
        if timeout is None:
            await self._wait_avatar_join_task
            return
        await asyncio.wait_for(asyncio.shield(self._wait_avatar_join_task), timeout=timeout)

    async def aclose(self) -> None:
        try:
            output = self._audio_output
            if output is not None:
                await output.aclose()
                if self._audio_output is output:
                    self._audio_output = None
        finally:
            if self._room is not None and self._room.isconnected():
                job_ctx = get_job_context(required=False)
                if job_ctx is not None:
                    try:
                        from livekit import api

                        await job_ctx.api.room.remove_participant(
                            api.RoomParticipantIdentity(
                                room=self._room.name,
                                identity=self.avatar_identity,
                            )
                        )
                    except Exception as e:
                        if isinstance(e, TwirpError) and e.code == "not_found":
                            logger.debug(
                                "avatar participant not in room, skipping removal",
                                extra={"identity": self.avatar_identity},
                            )
                        else:
                            logger.warning(
                                "failed to remove avatar participant",
                                extra={"identity": self.avatar_identity},
                                exc_info=True,
                            )

            if self._agent_session:
                self._agent_session.off("conversation_item_added", self._on_conversation_item_added)
                self._agent_session = None

            self._room = None
            if self._wait_avatar_join_task:
                await utils.aio.cancel_and_wait(self._wait_avatar_join_task)
                self._wait_avatar_join_task = None

    async def _wait_avatar_join(self, room: rtc.Room) -> None:
        # from the room's connection to the avatar's video track
        started_time, joined_time = await _core.wait_avatar_join(
            _core._core_create_room_session(Room.of(room)), self.avatar_identity
        )
        self._emit_metrics(
            AvatarMetrics(
                timestamp=joined_time,
                session_started_time=started_time,
                avatar_joined_time=joined_time,
                metadata=Metadata(
                    model_provider=self.provider,
                ),
            )
        )

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            playback_latency = ev.item.metrics.get("playback_latency")
            if playback_latency is not None:
                self._emit_metrics(
                    AvatarMetrics(
                        timestamp=ev.created_at,
                        playback_latency=playback_latency,
                        metadata=Metadata(
                            model_provider=self.provider,
                        ),
                    )
                )

    def _emit_metrics(self, metrics: AvatarMetrics) -> None:
        assert self._agent_session is not None
        if (core := self._agent_session._core) is not None:
            core.report_avatar_metrics(metrics)
        self.emit("metrics_collected", metrics)


RPC_CLEAR_BUFFER = "lk.clear_buffer"


RPC_PLAYBACK_FINISHED = "lk.playback_finished"


RPC_PLAYBACK_STARTED = "lk.playback_started"


AUDIO_STREAM_TOPIC = "lk.audio_stream"


# The agent's side: the session's audio sink, streaming to the avatar and paced by its reports.
class DataStreamAudioOutput(_core.DataStreamAudioOutput):
    """Audio sink sending speech to a remote avatar worker."""

    def __new__(
        cls,
        room: rtc.Room,
        *,
        destination_identity: str,
        sample_rate: int | None = None,
        wait_remote_track: int | None = None,
        clear_buffer_timeout: float | None = 2.0,
        wait_playback_start: bool = False,
    ) -> DataStreamAudioOutput:
        return super().__new__(
            cls,
            Room.of(room),
            destination_identity=destination_identity,
            sample_rate=sample_rate,
            wait_remote_track=wait_remote_track,
            clear_buffer_timeout=clear_buffer_timeout,
            wait_playback_start=wait_playback_start,
        )


class DataStreamAudioReceiver(AudioReceiver):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    If the sender_identity is provided, subscribe to the specified participant. If not provided,
    subscribe to the first agent participant in the room.
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
            Room.of(room),
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

    async def _core_next_audio(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment | None, rtc.AudioFrame | AudioSegmentEnd | None]:
        return await self._native.next_audio()

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        _, _, _, item = await self._native.next_audio()
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._native.aclose()


class QueueAudioOutput(
    AudioOutput[Literal["clear_buffer"]],
    AudioReceiver[Literal["playback_finished", "playback_started", "playback_progressed"]],
):
    """Audio output and receiver backed by the native segment queue."""

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
        self._apply(self._native.capture(frame, key, self._core_audio_epoch))

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

    async def _core_next_audio(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment | None, rtc.AudioFrame | AudioSegmentEnd | None]:
        return await self._native.next_audio()

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        _, _, _, item = await self._native.next_audio()
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._apply(self._native.close())


class AvatarRunner:
    """Worker that generates synchronized avatar video based on received audio"""

    def __init__(
        self,
        room: rtc.Room,
        *,
        audio_recv: AudioReceiver,
        video_gen: VideoGenerator,
        options: AvatarOptions,
        _queue_size_ms: int = 100,
        # queue size of the AV synchronizer
        _lazy_publish: bool = True,
        # publish video and audio tracks until the first frame pushed
    ) -> None:
        self._audio_recv = audio_recv
        self._core = _core.AvatarRunner(
            _core._core_create_room_session(Room.of(room)),
            receiver=audio_recv,
            generator=video_gen,
            options=options,
            queue_size_ms=_queue_size_ms,
            lazy_publish=_lazy_publish,
        )

    @property
    def av_sync(self) -> _core.AVSynchronizer:
        return self._core.av_sync

    async def start(self) -> None:
        """Start the worker"""
        await self._audio_recv.start()
        self._audio_recv.on("clear_buffer", self._on_clear_buffer)
        await self._core.start()

    def _on_clear_buffer(self) -> None:
        self._audio_recv._core_audio_epoch += 1
        self._core.clear_buffer(self._audio_recv._core_audio_epoch)

    async def wait_for_complete(self) -> None:
        await self._core.wait_for_complete()

    async def aclose(self) -> None:
        try:
            await self._core.aclose()
        finally:
            self._audio_recv.off("clear_buffer", self._on_clear_buffer)
