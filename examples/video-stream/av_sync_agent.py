import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable

import av
import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, utils
from livekit.agents.utils.av_sync import AVSynchronizer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class MediaInfo:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int


class MediaFileStreamer:
    """Streams video and audio frames from a media file."""

    def __init__(self, media_file: str | Path) -> None:
        self._media_file = str(media_file)
        self._container = av.open(self._media_file)

        self._video_stream = self._container.streams.video[0]
        self._audio_stream = self._container.streams.audio[0]

        # Cache media info
        self._info = MediaInfo(
            video_width=self._video_stream.width,
            video_height=self._video_stream.height,
            video_fps=float(self._video_stream.average_rate),
            audio_sample_rate=self._audio_stream.sample_rate,
            audio_channels=self._audio_stream.channels,
        )

    @property
    def info(self) -> MediaInfo:
        return self._info

    async def stream_video(self) -> AsyncIterable[rtc.VideoFrame]:
        """Streams video frames from the media file."""
        container = av.open(self._media_file)
        try:
            for frame in container.decode(video=0):
                # Convert video frame to RGBA
                frame = frame.to_rgb().to_ndarray()
                frame_rgba = np.ones(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8
                )
                frame_rgba[:, :, :3] = frame
                yield rtc.VideoFrame(
                    width=frame.shape[1],
                    height=frame.shape[0],
                    type=rtc.VideoBufferType.RGBA,
                    data=frame_rgba.tobytes(),
                )
        finally:
            container.close()

    async def stream_audio(self) -> AsyncIterable[rtc.AudioFrame]:
        """Streams audio frames from the media file."""
        container = av.open(self._media_file)
        try:
            for frame in container.decode(audio=0):
                # Convert audio frame to raw int16 samples
                frame: np.ndarray = frame.to_ndarray(format="s16")
                frame = (frame * 32768).astype(np.int16)
                yield rtc.AudioFrame(
                    data=frame.tobytes(),
                    sample_rate=self.info.audio_sample_rate,
                    num_channels=frame.shape[0],
                    samples_per_channel=frame.shape[1],
                )
        finally:
            container.close()

    async def aclose(self) -> None:
        """Closes the media container."""
        self._container.close()


async def entrypoint(job: JobContext):
    await job.connect()
    room = job.room

    # Create media streamer
    media_path = "/path/to/sample.mp4"
    streamer = MediaFileStreamer(media_path)
    media_info = streamer.info

    # Create video and audio sources/tracks
    queue_size_ms = 100
    video_source = rtc.VideoSource(
        width=media_info.video_width,
        height=media_info.video_height,
    )
    audio_source = rtc.AudioSource(
        sample_rate=media_info.audio_sample_rate,
        num_channels=media_info.audio_channels,
        queue_size_ms=queue_size_ms,
    )

    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)

    # Publish tracks
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    await room.local_participant.publish_track(video_track, video_options)
    await room.local_participant.publish_track(audio_track, audio_options)

    # Create AV synchronizer
    av_sync = AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=media_info.video_fps,
        video_queue_size_ms=queue_size_ms,
    )

    @utils.log_exceptions(logger=logger)
    async def _push_video_frames(
        video_stream: AsyncIterable[rtc.VideoFrame], av_sync: AVSynchronizer
    ) -> None:
        """Task to push video frames to the AV synchronizer."""
        async for frame in video_stream:
            await av_sync.push(frame)

    @utils.log_exceptions(logger=logger)
    async def _push_audio_frames(
        audio_stream: AsyncIterable[rtc.AudioFrame], av_sync: AVSynchronizer
    ) -> None:
        """Task to push audio frames to the AV synchronizer."""
        async for frame in audio_stream:
            await av_sync.push(frame)

    try:
        while True:
            # Create and run video and audio streaming tasks
            video_stream = streamer.stream_video()
            audio_stream = streamer.stream_audio()

            video_task = asyncio.create_task(_push_video_frames(video_stream, av_sync))
            audio_task = asyncio.create_task(_push_audio_frames(audio_stream, av_sync))

            # Wait for both tasks to complete
            # TODO: wait the frame in buffer to be processed
            await asyncio.gather(video_task, audio_task)
    finally:
        await av_sync.aclose()
        await streamer.aclose()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=400,
        )
    )
