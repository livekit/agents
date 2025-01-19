import asyncio
import logging
from typing import Optional

from livekit import rtc
from livekit.agents import utils

from .io import AudioEndSentinel, AudioReceiver
from .videogen_example import MediaInfo, video_generator

logger = logging.getLogger(__name__)


class AvatarWorker:
    """Worker that generates synchronized avatar video based on received audio"""

    def __init__(
        self, room: rtc.Room, media_info: MediaInfo, queue_size_ms: int = 100
    ) -> None:
        self._room = room
        self._media_info = media_info
        self._queue_size_ms = queue_size_ms

        self._audio_receiver = AudioReceiver(room)
        self._audio_queue: asyncio.Queue[rtc.AudioFrame | AudioEndSentinel] = (
            asyncio.Queue()
        )
        self._audio_stream_received: asyncio.Event = asyncio.Event()
        self._interrupted = False

        # Audio/video sources
        self._audio_source = rtc.AudioSource(
            sample_rate=media_info.audio_sample_rate,
            num_channels=media_info.audio_channels,
            queue_size_ms=queue_size_ms,
        )
        self._video_source = rtc.VideoSource(
            width=media_info.video_width, height=media_info.video_height
        )
        # AV synchronizer
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=media_info.video_fps,
            video_queue_size_ms=queue_size_ms,
        )
        self._main_atask: Optional[asyncio.Task[None]] = None

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
        self._main_atask = asyncio.create_task(self._main_task())
        self._audio_receive_atask = asyncio.create_task(self._read_audio())

    def _handle_interrupt(self) -> None:
        # clear the audio queue
        self._interrupted = True
        while not self._audio_queue.empty():
            # TODO: this may clear the audio end sentinel,
            # do we still need a playback finished event for interrupted audios?
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _read_audio(self) -> None:
        async for frame in self._audio_receiver.stream():
            self._audio_queue.put_nowait(frame)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Process audio frames and generate synchronized video"""

        playback_position = 0.0
        async for frame in video_generator(
            self._media_info, self._audio_queue, self._av_sync
        ):
            if isinstance(frame, AudioEndSentinel):
                self._audio_receiver.notify_playback_finished(
                    playback_position=playback_position,
                    interrupted=self._interrupted,
                )
                self._interrupted = False
                playback_position = 0.0
                continue

            self._av_sync.push(frame.video_frame)
            if frame.audio_frame:
                self._av_sync.push(frame.audio_frame)
                playback_position += frame.audio_frame.duration

    async def aclose(self) -> None:
        await utils.aio.gracefully_cancel(self._main_atask)
        await utils.aio.gracefully_cancel(self._audio_receive_atask)
        await self._av_sync.aclose()
        await self._audio_source.aclose()
        await self._video_source.aclose()
