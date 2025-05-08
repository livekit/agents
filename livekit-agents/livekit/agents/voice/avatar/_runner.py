from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

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
    ) -> None:
        self._room = room
        self._video_gen = video_gen
        self._options = options
        self._queue_size_ms = _queue_size_ms

        self._audio_recv = audio_recv
        self._playback_position = 0.0
        self._audio_playing = False
        self._tasks: set[asyncio.Task] = set()

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
        self._publish_video_atask: asyncio.Task[None] | None = None

    @property
    def av_sync(self) -> rtc.AVSynchronizer:
        return self._av_sync

    async def start(self) -> None:
        """Start the worker"""

        # Start audio receiver
        await self._audio_recv.start()

        def _on_clear_buffer():
            task = asyncio.create_task(self._handle_clear_buffer())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._audio_recv.on("clear_buffer", _on_clear_buffer)

        # Publish tracks
        audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._audio_source)
        video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._video_source)
        audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        self._avatar_audio_publication = await self._room.local_participant.publish_track(
            audio_track, audio_options
        )
        self._avatar_video_publication = await self._room.local_participant.publish_track(
            video_track, video_options
        )

        # Start processing
        self._read_audio_atask = asyncio.create_task(self._read_audio())
        self._publish_video_atask = asyncio.create_task(self._publish_video())

    async def wait_for_subscription(self) -> None:
        await asyncio.gather(
            self._avatar_audio_publication.wait_for_subscription(),
            self._avatar_video_publication.wait_for_subscription(),
        )

    async def _read_audio(self) -> None:
        async for frame in self._audio_recv:
            if not self._audio_playing and isinstance(frame, rtc.AudioFrame):
                self._audio_playing = True
            await self._video_gen.push_audio(frame)

    @log_exceptions(logger=logger)
    async def _publish_video(self) -> None:
        """Process audio frames and generate synchronized video"""

        async for frame in self._video_gen:
            if isinstance(frame, AudioSegmentEnd):
                # notify the agent that the audio has finished playing
                if self._audio_playing:
                    notify_task = self._audio_recv.notify_playback_finished(
                        playback_position=self._playback_position,
                        interrupted=False,
                    )
                    if asyncio.iscoroutine(notify_task):
                        await notify_task
                    self._playback_position = 0.0
                self._audio_playing = False
                continue

            await self._av_sync.push(frame)
            if isinstance(frame, rtc.AudioFrame):
                self._playback_position += frame.duration

    async def _handle_clear_buffer(self) -> None:
        """Handle clearing the buffer and notify about interrupted playback"""
        tasks = []
        clear_result = self._video_gen.clear_buffer()
        if asyncio.iscoroutine(clear_result):
            tasks.append(clear_result)

        if self._audio_playing:
            notify_task = self._audio_recv.notify_playback_finished(
                playback_position=self._playback_position,
                interrupted=True,
            )
            if asyncio.iscoroutine(notify_task):
                tasks.append(notify_task)
            self._playback_position = 0.0
            self._audio_playing = False

        await asyncio.gather(*tasks)

    async def aclose(self) -> None:
        if self._publish_video_atask:
            await aio.cancel_and_wait(self._publish_video_atask)
        if self._read_audio_atask:
            await aio.cancel_and_wait(self._read_audio_atask)
        await aio.cancel_and_wait(*self._tasks)

        await self._av_sync.aclose()
        await self._audio_source.aclose()
        await self._video_source.aclose()
