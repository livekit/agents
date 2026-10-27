"""Avatars: the session and its video generator, the runner an avatar worker drives, the queue
audio output, and the agent and an avatar worker exchanging audio over LiveKit DataStream: one
byte stream per segment on ``lk.audio_stream``, ``lk.clear_buffer`` from the agent,
``lk.playback_started`` and ``lk.playback_finished`` back. Both DataStream ends run in the core
(``_core.DataStreamAudioOutput``, ``_core.DataStreamAudioReceiver``)."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from livekit import rtc
from livekit.api import TwirpError

from .. import _core, utils
from ..job import get_job_context
from ..metrics import AvatarMetrics, Metadata
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import aio, log_exceptions
from .events import ConversationItemAddedEvent
from .io import AudioOutput, AudioOutputCapabilities

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


class AudioReceiver(ABC, rtc.EventEmitter[Literal["clear_buffer"]]):
    async def start(self) -> None:
        pass

    @abstractmethod
    def notify_playback_finished(
        self, playback_position: float, interrupted: bool
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has finished"""

    @abstractmethod
    def notify_playback_started(self) -> None | Coroutine[None, None, None]:
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


TEvent = TypeVar("TEvent")


class AvatarSession(ABC, rtc.EventEmitter[Literal["metrics_collected"] | TEvent], Generic[TEvent]):
    """Base class for avatar plugin sessions."""

    def __init__(self) -> None:
        super().__init__()
        self._wait_avatar_join_task: asyncio.Task[None] | None = None
        self._room: rtc.Room | None = None
        self._agent_session: VoiceSession | None = None

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
        started_time, joined_time = await _core.wait_avatar_join(room, self.avatar_identity)
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
            room,
            sender_identity=sender_identity,
            frame_size_ms=frame_size_ms or 100,
            rpc_max_retries=rpc_max_retries,
            on_clear_buffer=lambda: self.emit("clear_buffer"),
        )

    async def start(self) -> None:
        await self._native.start()

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self._native.notify_playback_finished(playback_position, interrupted)

    def notify_playback_started(self) -> None:
        self._native.notify_playback_started()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        frame = await self._native.recv()
        return AudioSegmentEnd() if frame is None else frame

    async def aclose(self) -> None:
        await self._native.aclose()


class QueueAudioOutput(
    AudioOutput,
    AudioReceiver,
    rtc.EventEmitter[Literal["clear_buffer"]],
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
        self._data_ch = utils.aio.Chan[rtc.AudioFrame | AudioSegmentEnd]()
        self._capturing = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and queue audio frame"""
        await super().capture_frame(frame)
        if not self._capturing:
            self._capturing = True
            if not self._wait_playback_start:
                self.on_playback_started(created_at=time.time())

        await self._data_ch.send(frame)

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if not self._capturing:
            return
        self._capturing = False
        self._data_ch.send_nowait(AudioSegmentEnd())

    # as AudioReceiver for AvatarRunner

    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        while True:
            try:
                self._data_ch.recv_nowait()
            except utils.aio.ChanEmpty:
                break
        self.emit("clear_buffer")  # type: ignore

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self.on_playback_finished(playback_position=playback_position, interrupted=interrupted)

    def notify_playback_started(self) -> None:
        if not self._wait_playback_start:
            # already fired eagerly in capture_frame; avoid double-firing
            return
        self.on_playback_started(created_at=time.time())

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self._data_ch


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
        # what played of the segment, and whether one is playing (the core's bookkeeping)
        self._playback = _core.AvatarPlayback()
        self._tasks: set[asyncio.Task[Any]] = set()

        self._lock = asyncio.Lock()
        self._audio_publication: rtc.LocalTrackPublication | None = None
        self._video_publication: rtc.LocalTrackPublication | None = None
        self._lazy_publish = _lazy_publish

        # Audio/video sources
        self._audio_source = rtc.AudioSource(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            queue_size_ms=self._queue_size_ms,
        )
        self._video_source = rtc.VideoSource(width=options.video_width, height=options.video_height)
        # AV synchronizer
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=options.video_fps,
            video_queue_size_ms=self._queue_size_ms,
        )
        self._read_audio_task: asyncio.Task[None] | None = None
        self._forward_video_atask: asyncio.Task[None] | None = None
        self._room_connected_fut = asyncio.Future[None]()

    @property
    def av_sync(self) -> rtc.AVSynchronizer:
        return self._av_sync

    async def start(self) -> None:
        """Start the worker"""

        # start audio receiver
        await self._audio_recv.start()
        self._audio_recv.on("clear_buffer", self._on_clear_buffer)

        self._room.on("connection_state_changed", self._on_connection_state_changed)
        if self._room.isconnected():
            self._room_connected_fut.set_result(None)

        if not self._lazy_publish:
            await self._publish_track()

        # start processing
        self._read_audio_atask = asyncio.create_task(self._read_audio())
        self._forward_video_atask = asyncio.create_task(self._forward_video())

    async def wait_for_complete(self) -> None:
        if not self._read_audio_atask or not self._forward_video_atask:
            raise RuntimeError("AvatarRunner not started")

        await asyncio.gather(
            self._read_audio_atask,
            self._forward_video_atask,
        )

    async def _publish_track(self) -> None:
        async with self._lock:
            await self._room_connected_fut

            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self._audio_publication = await self._room.local_participant.publish_track(
                audio_track, audio_options
            )
            await self._audio_publication.wait_for_subscription()

            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._video_source)
            video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            self._video_publication = await self._room.local_participant.publish_track(
                video_track, video_options
            )

    @log_exceptions(logger=logger)
    async def _read_audio(self) -> None:
        async for frame in self._audio_recv:
            if isinstance(frame, rtc.AudioFrame):
                self._playback.audio_received()
            await self._video_gen.push_audio(frame)

    @log_exceptions(logger=logger)
    async def _forward_video(self) -> None:
        """Forward video to the room through the AV synchronizer"""

        async for frame in self._video_gen:
            if isinstance(frame, AudioSegmentEnd):
                # notify the agent that the audio has finished playing
                if (position := self._playback.segment_ended()) is not None:
                    self._notify(
                        self._audio_recv.notify_playback_finished(
                            playback_position=position, interrupted=False
                        )
                    )
                continue

            if not self._video_publication:
                await self._publish_track()

            await self._av_sync.push(frame)
            if isinstance(frame, rtc.AudioFrame) and self._playback.audio_played(frame.duration):
                self._notify(self._audio_recv.notify_playback_started())

    def _notify(self, notify_task: Any) -> None:
        # a receiver may report asynchronously: never block the video forwarding on it
        if asyncio.iscoroutine(notify_task):
            task = asyncio.create_task(notify_task)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _on_clear_buffer(self) -> None:
        """Handle clearing the buffer and notify about interrupted playback"""

        @log_exceptions(logger=logger)
        async def _handle_clear_buffer(audio_playing: bool) -> None:
            clear_task = self._video_gen.clear_buffer()
            if asyncio.iscoroutine(clear_task):
                await clear_task

            if audio_playing:
                notify_task = self._audio_recv.notify_playback_finished(
                    playback_position=self._playback.cleared(), interrupted=True
                )
                if asyncio.iscoroutine(notify_task):
                    await notify_task

        task = asyncio.create_task(_handle_clear_buffer(self._playback.clear()))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_connection_state_changed(self, _: rtc.ConnectionState) -> None:
        if self._room.isconnected() and not self._room_connected_fut.done():
            self._room_connected_fut.set_result(None)

    async def aclose(self) -> None:
        self._room.off("connection_state_changed", self._on_connection_state_changed)

        await self._audio_recv.aclose()
        if self._forward_video_atask:
            await aio.cancel_and_wait(self._forward_video_atask)
        if self._read_audio_atask:
            await aio.cancel_and_wait(self._read_audio_atask)
        await aio.cancel_and_wait(*self._tasks)

        await self._av_sync.aclose()
        await self._audio_source.aclose()
        await self._video_source.aclose()
