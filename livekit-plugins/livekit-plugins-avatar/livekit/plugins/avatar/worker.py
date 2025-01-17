import asyncio
from dataclasses import dataclass
from typing import Optional

from livekit import rtc
from livekit.agents.pipeline.io import PlaybackFinishedEvent

from .sink import DataStreamAudioReceiver


@dataclass
class MediaInfo:
    """Configuration for media streams"""

    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int


class AvatarWorker:
    """Worker that generates synchronized avatar video based on received audio"""

    def __init__(
        self,
        room: rtc.Room,
        media_info: MediaInfo,
        queue_size_ms: int = 50,
    ) -> None:
        self._room = room
        self._media_info = media_info
        self._queue_size_ms = queue_size_ms

        # Audio/video sources
        self._audio_source = rtc.AudioSource(
            sample_rate=media_info.audio_sample_rate,
            num_channels=media_info.audio_channels,
            queue_size_ms=queue_size_ms,
        )
        self._video_source = rtc.VideoSource(
            width=media_info.video_width,
            height=media_info.video_height,
        )

        # Tracks for publishing
        self._audio_track = rtc.LocalAudioTrack.create_audio_track(
            "avatar_audio", self._audio_source
        )
        self._video_track = rtc.LocalVideoTrack.create_video_track(
            "avatar_video", self._video_source
        )

        # Audio receiver
        self._audio_receiver = DataStreamAudioReceiver(room)

        # AV synchronizer
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=media_info.video_fps,
            video_queue_size_ms=queue_size_ms,
        )

        self._current_position: float = 0.0

    async def start(self) -> None:
        """Start the worker"""
        # Start audio receiver
        await self._audio_receiver.start()

        # Publish tracks
        audio_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await self._room.local_participant.publish_track(
            self._audio_track, audio_options
        )
        await self._room.local_participant.publish_track(
            self._video_track, video_options
        )

        # Handle interrupts
        self._audio_receiver.on(
            "playback_interrupted",
            lambda: asyncio.create_task(self._handle_interrupt()),
        )

        # Start processing
        asyncio.create_task(self._process_audio())

    async def _handle_interrupt(self) -> None:
        """Handle playback interrupt from sink"""
        await self._audio_receiver.control.notify_playback_finished(
            PlaybackFinishedEvent(
                playback_position=self._current_position,
                interrupted=True,
            )
        )
        self._current_position = 0.0

    async def _process_audio(self) -> None:
        """Process audio frames and generate synchronized video"""
        try:
            async for frame in self._audio_receiver.receive():
                # Update position
                self._current_position += frame.duration

                # Generate video frame
                video_frame = self._generate_video_frame(frame)

                # Push frames to synchronizer
                await self._av_sync.push(frame)
                if video_frame:
                    await self._av_sync.push(video_frame)

            # Normal end of segment
            await self._audio_receiver.control.notify_playback_finished(
                PlaybackFinishedEvent(
                    playback_position=self._current_position,
                    interrupted=False,
                )
            )
            self._current_position = 0.0

        except asyncio.CancelledError:
            raise
        finally:
            # Cleanup
            await self._av_sync.aclose()
            await self._audio_source.aclose()
            await self._video_source.aclose()

    def _generate_video_frame(
        self, audio_frame: rtc.AudioFrame
    ) -> Optional[rtc.VideoFrame]:
        """Generate avatar video frame based on audio frame"""
        # TODO: Implement avatar video generation
        # For now, return None to skip video generation
        return None

    async def close(self) -> None:
        """Close the worker and cleanup resources"""
        await self._audio_receiver.close()
        if self._av_sync:
            await self._av_sync.aclose()
        if self._audio_source:
            await self._audio_source.aclose()
        if self._video_source:
            await self._video_source.aclose()
