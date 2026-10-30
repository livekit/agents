"""Avatars: the session and its video generator, the runner an avatar worker drives, the queue
audio output, and the agent and an avatar worker exchanging audio over LiveKit DataStream: one
byte stream per segment on ``lk.audio_stream``, ``lk.clear_buffer`` from the agent,
``lk.playback_started`` and ``lk.playback_finished`` back. Both DataStream ends run in the core
(``_core.DataStreamAudioOutput``, ``_core.DataStreamAudioReceiver``), as does the runner
(``_core.AvatarRunner``): this module keeps the SDK's synchronizer and tracks it pushes into."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import rtc
from livekit.api import TwirpError

from .. import _core, utils
from .._generated.io_events import AvatarPlaybackSegment
from ..job import get_job_context
from ..metrics import AvatarMetrics, Metadata
from ..types import NOT_GIVEN, NotGivenOr
from .events import ConversationItemAddedEvent
from .io import AudioOutput, AudioOutputCapabilities, PlaybackSegment
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


class AudioSegmentEnd:
    pass


_ReceiverEventT = TypeVar("_ReceiverEventT", default=Never, contravariant=True)


class AudioReceiver(
    ABC, rtc.EventEmitter[Literal["clear_buffer"] | _ReceiverEventT], Generic[_ReceiverEventT]
):
    _core_audio_epoch = 0
    _core_reading = False

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
        self._core_reading = False

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
DataStreamAudioOutput = _core.DataStreamAudioOutput


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


@dataclass
class _QueuedAudioSegment:
    epoch: int
    key: PlaybackSegment
    yielded: bool = False
    started: bool = False
    discarded: bool = False


class QueueAudioOutput(
    AudioOutput[Literal["clear_buffer"]],
    AudioReceiver[Literal["playback_finished", "playback_started", "playback_progressed"]],
):
    """
    AudioOutput implementation that sends audio frames through a queue.
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        wait_playback_start: bool = False,
    ):
        super().__init__(
            label="DebugQueueIO",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self._wait_playback_start = wait_playback_start
        self._data_ch = utils.aio.Chan[
            tuple[_QueuedAudioSegment, rtc.AudioFrame | AudioSegmentEnd]
        ]()
        self._segments: deque[_QueuedAudioSegment] = deque()
        self._capture_segment: _QueuedAudioSegment | None = None
        self._clear_requested = True

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and queue audio frame"""
        if self._data_ch.closed:
            raise utils.aio.ChanClosed
        await super().capture_frame(frame)
        self._clear_requested = False
        if self._capture_segment is None:
            key = self.playback_segment
            assert key is not None
            self._capture_segment = _QueuedAudioSegment(self._core_audio_epoch, key)
            self._segments.append(self._capture_segment)
            if not self._wait_playback_start:
                self.on_playback_started(segment=self._capture_segment.key, created_at=time.time())

        await self._data_ch.send((self._capture_segment, frame))

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if self._capture_segment is None:
            return
        self._data_ch.send_nowait((self._capture_segment, AudioSegmentEnd()))
        self._capture_segment = None

    # as AudioReceiver for AvatarRunner

    def clear_buffer(self) -> None:
        """Discard queued audio; its playback report still settles the interrupted segment."""
        if self._data_ch.closed:
            return
        self._capture_segment = None
        self._tracker.note_flush()
        for segment in self._segments:
            if not segment.yielded:
                segment.discarded = True
        while True:
            try:
                self._data_ch.recv_nowait()
            except utils.aio.ChanEmpty:
                break
        self._finish_discarded()
        if not self._clear_requested:
            self._clear_requested = True
            self.emit("clear_buffer")

    def _finish_discarded(self) -> None:
        # Unread audio cannot receive a consumer acknowledgement. It is always newer than
        # previously yielded segments, so settle it only after their reports arrive.
        while self._segments and self._segments[0].discarded:
            segment = self._segments.popleft()
            self.on_playback_finished(segment=segment.key, playback_position=0.0, interrupted=True)

    def notify_playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
        *,
        segment: AvatarPlaybackSegment | None = None,
    ) -> None:
        # A consumer reading the queue reports in delivery order: without a key, the report
        # belongs to the oldest delivered segment and settles the key captured for it.
        if (
            not self._segments
            or not self._segments[0].yielded
            or (segment is not None and self._segments[0].key != segment.playback)
        ):
            return
        settled = self._segments.popleft()
        self.on_playback_finished(
            segment=settled.key, playback_position=playback_position, interrupted=interrupted
        )
        self._finish_discarded()

    def notify_playback_started(self, *, segment: AvatarPlaybackSegment | None = None) -> None:
        if not self._wait_playback_start:
            return
        if segment is not None:
            self.on_playback_started(segment=segment.playback, created_at=time.time())
            return
        # without a key: the oldest delivered segment whose start was not reported yet
        queued = next((s for s in self._segments if s.yielded and not s.started), None)
        if queued is None:
            return
        queued.started = True
        self.on_playback_started(segment=queued.key, created_at=time.time())

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def _next_observation(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment, rtc.AudioFrame | AudioSegmentEnd]:
        segment, item = await anext(self._data_ch)
        starts_segment = isinstance(item, rtc.AudioFrame) and not segment.yielded
        if isinstance(item, rtc.AudioFrame):
            segment.yielded = True
        return (
            segment.epoch,
            starts_segment,
            AvatarPlaybackSegment(playback=segment.key, stream_id=None),
            item,
        )

    async def _core_next_audio(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment | None, rtc.AudioFrame | AudioSegmentEnd | None]:
        try:
            return await self._next_observation()
        except StopAsyncIteration:
            return self._core_audio_epoch, False, None, None

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        return (await self._next_observation())[3]

    async def aclose(self) -> None:
        if self._data_ch.closed:
            return
        self.clear_buffer()
        self._tracker.reset()
        self._segments.clear()
        self._data_ch.close()


@dataclass
class AvatarOptions:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int


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
        self._room = room
        self._video_gen = video_gen
        self._options = options
        self._queue_size_ms = _queue_size_ms
        self._audio_recv = audio_recv
        self._media = _AvatarMedia(room, options, _queue_size_ms)
        self._core = _core.AvatarRunner(
            _core._core_create_room_session(Room.of(room)),
            receiver=audio_recv,
            generator=video_gen,
            media=self._media,
            lazy_publish=_lazy_publish,
        )

    @property
    def av_sync(self) -> rtc.AVSynchronizer:
        return self._media.av_sync

    async def start(self) -> None:
        """Start the worker"""
        await self._audio_recv.start()
        self._audio_recv.on("clear_buffer", self._on_clear_buffer)
        await self._core.start()

    def _on_clear_buffer(self) -> None:
        self._audio_recv._core_audio_epoch += 1
        self._audio_recv._core_reading = False
        self._core.clear_buffer(self._audio_recv._core_audio_epoch)

    async def wait_for_complete(self) -> None:
        await self._core.wait_for_complete()

    async def aclose(self) -> None:
        try:
            await self._core.aclose()
        finally:
            self._audio_recv.off("clear_buffer", self._on_clear_buffer)


class _AvatarMedia:
    """What the core's runner pushes into: the avatar's tracks and its AV synchronizer."""

    def __init__(self, room: rtc.Room, options: AvatarOptions, queue_size_ms: int) -> None:
        self._room = room
        self.audio_source = rtc.AudioSource(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            queue_size_ms=queue_size_ms,
        )
        self.video_source = rtc.VideoSource(width=options.video_width, height=options.video_height)
        self.av_sync = rtc.AVSynchronizer(
            audio_source=self.audio_source,
            video_source=self.video_source,
            video_fps=options.video_fps,
            video_queue_size_ms=queue_size_ms,
        )

    async def _core_publish(self) -> None:
        audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self.audio_source)
        audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        publication = await self._room.local_participant.publish_track(audio_track, audio_options)
        await publication.wait_for_subscription()

        video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self.video_source)
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await self._room.local_participant.publish_track(video_track, video_options)

    async def _core_push(self, frame: rtc.VideoFrame | rtc.AudioFrame) -> None:
        await self.av_sync.push(frame)

    async def _core_aclose(self) -> None:
        await self.av_sync.aclose()
        await self.audio_source.aclose()
        await self.video_source.aclose()
