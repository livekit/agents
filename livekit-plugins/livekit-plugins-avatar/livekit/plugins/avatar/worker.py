import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol

from livekit import rtc
from livekit.agents import utils

from .io import AudioFlushSentinel, AudioReceiver

logger = logging.getLogger(__name__)


class VideoGenerator(Protocol):
    async def push_audio(self, frame: rtc.AudioFrame | AudioFlushSentinel) -> None:
        """Push an audio frame to the video generator"""

    def clear_buffer(self) -> None:
        """Clear the audio buffer, stopping audio playback immediately"""

    async def stream(
        self,
    ) -> AsyncIterator[
        tuple[rtc.VideoFrame, Optional[rtc.AudioFrame]] | AudioFlushSentinel
    ]:
        """Continuously yield video frames, idle frames are yielded when no audio is available"""


@dataclass
class MediaOptions:
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
        *,
        video_generator: VideoGenerator,
        media_options: MediaOptions,
        _queue_size_ms: int = 100,
    ) -> None:
        self._room = room
        self._video_generator = video_generator
        self._media_options = media_options
        self._queue_size_ms = _queue_size_ms

        self._audio_receiver = AudioReceiver(room)
        self._audio_stream_received: asyncio.Event = asyncio.Event()
        self._interrupted = False

        # Audio/video sources
        self._audio_source = rtc.AudioSource(
            sample_rate=media_options.audio_sample_rate,
            num_channels=media_options.audio_channels,
            queue_size_ms=self._queue_size_ms,
        )
        self._video_source = rtc.VideoSource(
            width=media_options.video_width, height=media_options.video_height
        )
        # AV synchronizer
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=media_options.video_fps,
            video_queue_size_ms=self._queue_size_ms,
        )
        self._video_gen_atask: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the worker"""

        # Start audio receiver
        await self._audio_receiver.start()
        self._audio_receiver.on("interrupt_playback", self._handle_interrupt)

        # Publish tracks
        audio_track = rtc.LocalAudioTrack.create_audio_track(
            "avatar_audio", self._audio_source
        )
        video_track = rtc.LocalVideoTrack.create_video_track(
            "avatar_video", self._video_source
        )
        audio_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await self._room.local_participant.publish_track(audio_track, audio_options)
        await self._room.local_participant.publish_track(video_track, video_options)

        # Start processing
        self._audio_receive_atask = asyncio.create_task(self._read_audio())
        self._video_gen_atask = asyncio.create_task(self._stream_video())

    async def _read_audio(self) -> None:
        async for frame in self._audio_receiver.stream():
            self._video_generator.push_audio(frame)

    @utils.log_exceptions(logger=logger)
    async def _stream_video(self) -> None:
        """Process audio frames and generate synchronized video"""

        playback_position = 0.0
        video_stream = await self._video_generator.stream()
        async for frame in video_stream:
            if isinstance(frame, AudioFlushSentinel):
                # notify the agent that the audio has finished playing
                self._audio_receiver.notify_playback_finished(
                    playback_position=playback_position,
                    interrupted=self._interrupted,
                )
                self._interrupted = False
                playback_position = 0.0
                continue

            video_frame, audio_frame = frame
            self._av_sync.push(video_frame)
            if audio_frame:
                self._av_sync.push(audio_frame)
                playback_position += audio_frame.duration

    def _handle_interrupt(self) -> None:
        # clear the audio queue
        self._interrupted = True
        self._video_generator.clear_buffer()
        # TODO: this may clear the audio end sentinel,
        # do we still need a playback finished event for interrupted audios?

    async def aclose(self) -> None:
        await utils.aio.gracefully_cancel(self._video_gen_atask)
        await utils.aio.gracefully_cancel(self._audio_receive_atask)
        await self._av_sync.aclose()
        await self._audio_source.aclose()
        await self._video_source.aclose()
