from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from livekit import rtc

from ...utils import aio, log_exceptions
from ._types import AudioReceiver, AudioSegmentEnd, VideoGenerator

logger = logging.getLogger(__name__)


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
        self._playback_position = 0.0
        self._audio_playing = False
        self._tasks: set[asyncio.Task[Any]] = set()

        self._lock = asyncio.Lock()
        self._audio_publication: rtc.LocalTrackPublication | None = None
        self._video_publication: rtc.LocalTrackPublication | None = None
        self._republish_atask: asyncio.Task[None] | None = None
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

        self._room.on("reconnected", self._on_reconnected)
        self._room.on("connection_state_changed", self._on_connection_state_changed)
        if self._room.isconnected():
            self._room_connected_fut.set_result(None)

        if not self._lazy_publish:
            await self._publish_track()

        # start processing
        self._read_audio_atask = asyncio.create_task(self._read_audio())
        self._forward_video_atask = asyncio.create_task(self._forward_video())

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
            if not self._audio_playing and isinstance(frame, rtc.AudioFrame):
                self._audio_playing = True
            await self._video_gen.push_audio(frame)

    @log_exceptions(logger=logger)
    async def _forward_video(self) -> None:
        """Forward video to the room through the AV synchronizer"""

        async for frame in self._video_gen:
            if isinstance(frame, AudioSegmentEnd):
                # notify the agent that the audio has finished playing
                if self._audio_playing:
                    notify_task = self._audio_recv.notify_playback_finished(
                        playback_position=self._playback_position,
                        interrupted=False,
                    )
                    self._audio_playing = False
                    self._playback_position = 0.0
                    if asyncio.iscoroutine(notify_task):
                        await notify_task
                continue

            if not self._video_publication:
                await self._publish_track()

            await self._av_sync.push(frame)
            if isinstance(frame, rtc.AudioFrame):
                self._playback_position += frame.duration

    def _on_clear_buffer(self) -> None:
        """Handle clearing the buffer and notify about interrupted playback"""

        @log_exceptions(logger=logger)
        async def _handle_clear_buffer(audio_playing: bool) -> None:
            clear_task = self._video_gen.clear_buffer()
            if asyncio.iscoroutine(clear_task):
                await clear_task

            if audio_playing:
                notify_task = self._audio_recv.notify_playback_finished(
                    playback_position=self._playback_position,
                    interrupted=True,
                )
                self._playback_position = 0.0
                if asyncio.iscoroutine(notify_task):
                    await notify_task

        task = asyncio.create_task(_handle_clear_buffer(self._audio_playing))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._audio_playing = False

    def _on_reconnected(self) -> None:
        if self._lazy_publish and not self._video_publication:
            return

        if self._republish_atask:
            self._republish_atask.cancel()
        self._republish_atask = asyncio.create_task(self._publish_track())

    def _on_connection_state_changed(self, _: rtc.ConnectionState) -> None:
        if self._room.isconnected() and not self._room_connected_fut.done():
            self._room_connected_fut.set_result(None)

    async def aclose(self) -> None:
        self._room.off("reconnected", self._on_reconnected)
        self._room.off("connection_state_changed", self._on_connection_state_changed)

        if self._forward_video_atask:
            await aio.cancel_and_wait(self._forward_video_atask)
        if self._read_audio_atask:
            await aio.cancel_and_wait(self._read_audio_atask)
        await aio.cancel_and_wait(*self._tasks)

        if self._republish_atask:
            await aio.cancel_and_wait(self._republish_atask)

        await self._av_sync.aclose()
        await self._audio_source.aclose()
        await self._video_source.aclose()
