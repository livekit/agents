import asyncio
from dataclasses import dataclass
from typing import Optional

from livekit import rtc

from ..utils import aio
from .io import AudioSink, AudioStream, VideoStream
from .log import logger


class RoomAudioSink(AudioSink):
    """AudioSink implementation that publishes audio to a LiveKit room"""

    def __init__(
        self, room: rtc.Room, *, sample_rate: int = 24000, num_channels: int = 1
    ) -> None:
        """Initialize the RoomAudioSink

        Args:
            room: The LiveKit room to publish audio to
            sample_rate: Sample rate of the audio in Hz
            num_channels: Number of audio channels
        """
        super().__init__(sample_rate=sample_rate)
        self._room = room

        # Create audio source and track
        self._audio_source = rtc.AudioSource(
            sample_rate=sample_rate, num_channels=num_channels
        )
        self._track = rtc.LocalAudioTrack.create_audio_track(
            "assistant_voice", self._audio_source
        )

        self._publication: rtc.LocalTrackPublication | None = None
        self._publish_task: asyncio.Task | None = None
        self._pushed_duration: float | None = None

    async def start(self) -> None:
        """Start publishing the audio track to the room"""
        if self._publication:
            return

        # TODO: handle reconnected, do we need to cancel and re-publish? seems no
        self._publication = await self._room.local_participant.publish_track(
            self._track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        # is this necessary?
        await self._publication.wait_for_subscription()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture an audio frame and publish it to the room"""
        await super().capture_frame(frame)

        if self._pushed_duration is None:
            self._pushed_duration = 0.0

        self._pushed_duration += frame.duration
        await self._audio_source.capture_frame(frame)

    def flush(self) -> None:
        """Flush the current audio segment and notify when complete"""
        super().flush()

        if self._pushed_duration is not None:
            self._notify_playback_finished(self._pushed_duration, interrupted=False)
            self._pushed_duration = None

    def clear_buffer(self) -> None:
        """Clear the audio buffer immediately"""
        self._audio_source.clear_queue()

        if self._pushed_duration is not None:
            self._notify_playback_finished(self._pushed_duration, interrupted=True)
            self._pushed_duration = None

    def _notify_playback_finished(
        self, playback_position: float, interrupted: bool
    ) -> None:
        """Wait for the audio to be played out and notify when complete"""
        playout_task = asyncio.create_task(self._audio_source.wait_for_playout())
        playout_task.add_done_callback(
            lambda _: self.on_playback_finished(
                playback_position=playback_position, interrupted=interrupted
            )
        )


@dataclass
class RoomInputOptions:
    subscribe_audio: bool = True
    """Whether to subscribe to audio"""
    subscribe_video: bool = False
    """Whether to subscribe to video"""
    audio_sample_rate: int = 16000
    """Sample rate of the input audio in Hz"""
    audio_num_channels: int = 1
    """Number of audio channels"""
    video_buffer_size: Optional[int] = 30
    """Buffer size of the video in number of frames, None means unlimited"""
    warn_dropped_video_frames: bool = True
    """Whether to warn when video frames are dropped"""


DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()


class RoomInput:
    """Creates video and audio streams from a remote participant in a LiveKit room"""

    class _RemoteTrackHandler:
        """Manages streaming from a remote track to a aio.Chan"""

        def __init__(self, options: RoomInputOptions) -> None:
            self._options = options

            self._remote_track: rtc.RemoteTrack | None = None
            self._main_atask: asyncio.Task | None = None

            self._data_ch: aio.Chan[rtc.AudioFrame | rtc.VideoFrame] = aio.Chan()
            self._enabled = True  # stream the frame or not

        @property
        def data_ch(self) -> aio.Chan[rtc.AudioFrame | rtc.VideoFrame]:
            return self._data_ch

        @property
        def enabled(self) -> bool:
            return self._enabled

        @enabled.setter
        def enabled(self, enabled: bool) -> None:
            """Drop frames if the stream is not enabled"""
            self._enabled = enabled

        def setup(self, track: rtc.RemoteTrack) -> None:
            """Set up streaming for a new track"""
            if track == self._remote_track:
                return

            if self._main_atask is not None:
                self._main_atask.cancel()

            self._remote_track = track
            if isinstance(track, rtc.RemoteAudioTrack):
                stream = rtc.AudioStream(
                    track=track,
                    sample_rate=self._options.audio_sample_rate,
                    num_channels=self._options.audio_num_channels,
                )
            elif isinstance(track, rtc.RemoteVideoTrack):
                stream = rtc.VideoStream(track=track)
            else:
                raise ValueError(f"Unsupported track source type: {type(track)}")

            self._main_atask = asyncio.create_task(self._read_stream(stream))

        async def _read_stream(self, stream: rtc.AudioStream | rtc.VideoStream) -> None:
            async for event in stream:
                if not self._enabled:
                    continue
                self._data_ch.send_nowait(event.frame)
                if (
                    isinstance(event.frame, rtc.VideoFrame)
                    and self._options.video_buffer_size is not None
                ):
                    dropped = 0
                    while self._data_ch.qsize() > self._options.video_buffer_size:
                        await self._data_ch.recv()  # drop old frames if buffer is full
                        dropped += 1
                    if dropped > 0 and self._options.warn_dropped_video_frames:
                        logger.warning(
                            "dropping video frames since buffer is full",
                            extra={
                                "buffer_size": self._options.video_buffer_size,
                                "dropped": dropped,
                            },
                        )

        async def aclose(self) -> None:
            if self._main_atask is not None:
                await aio.graceful_cancel(self._main_atask)
            self._main_atask = None
            self._remote_track = None
            self._data_ch.close()

    def __init__(
        self,
        room: rtc.Room,
        participant_identity: Optional[str] = None,
        options: RoomInputOptions = DEFAULT_ROOM_INPUT_OPTIONS,
    ) -> None:
        """
        Args:
            room: The LiveKit room to get streams from
            participant_identity: Optional identity of the participant to get streams from.
                                If None, will use the first participant that joins.
            options: RoomInputOptions
        """
        self._options = options
        self._room = room
        self._expected_identity = participant_identity
        self._participant: rtc.RemoteParticipant | None = None
        self._closed = False

        # set up track handlers
        self._track_handlers: dict[
            rtc.TrackSource.ValueType, RoomInput._RemoteTrackHandler
        ] = {}
        if self._options.subscribe_audio:
            self._track_handlers[rtc.TrackSource.SOURCE_MICROPHONE] = (
                self._RemoteTrackHandler(self._options)
            )
        if self._options.subscribe_video:
            self._track_handlers[rtc.TrackSource.SOURCE_CAMERA] = (
                self._RemoteTrackHandler(self._options)
            )

        self._participant_ready = asyncio.Event()
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("track_published", self._subscribe_to_tracks)
        self._room.on("track_subscribed", self._subscribe_to_tracks)
        self._room.on("reconnected", self._subscribe_to_tracks)

        # try to find participant
        if self._expected_identity is not None:
            participant = self._room.remote_participants.get(self._expected_identity)
            if participant is not None:
                self._link_participant(participant)
        else:
            for participant in self._room.remote_participants.values():
                self._link_participant(participant)
                if self._participant:
                    break

    async def wait_for_participant(self) -> rtc.RemoteParticipant:
        await self._participant_ready.wait()
        assert self._participant is not None
        return self._participant

    @property
    def audio(self) -> AudioStream | None:
        if rtc.TrackSource.SOURCE_MICROPHONE not in self._track_handlers:
            return None
        return self._track_handlers[rtc.TrackSource.SOURCE_MICROPHONE].data_ch

    @property
    def video(self) -> VideoStream | None:
        if rtc.TrackSource.SOURCE_CAMERA not in self._track_handlers:
            return None
        return self._track_handlers[rtc.TrackSource.SOURCE_CAMERA].data_ch

    @property
    def audio_enabled(self) -> bool:
        if rtc.TrackSource.SOURCE_MICROPHONE not in self._track_handlers:
            return False
        return self._track_handlers[rtc.TrackSource.SOURCE_MICROPHONE].enabled

    @property
    def video_enabled(self) -> bool:
        if rtc.TrackSource.SOURCE_CAMERA not in self._track_handlers:
            return False
        return self._track_handlers[rtc.TrackSource.SOURCE_CAMERA].enabled

    def set_enabled(self, source: rtc.TrackSource.ValueType, enabled: bool) -> None:
        if source not in self._track_handlers:
            raise ValueError(f"Track {source} is not subscribed")
        self._track_handlers[source].enabled = enabled

    def _link_participant(self, participant: rtc.RemoteParticipant) -> None:
        if (
            self._expected_identity is not None
            and participant.identity != self._expected_identity
        ):
            return

        self._participant = participant
        self._participant_ready.set()

        # set up tracks
        self._subscribe_to_tracks()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if self._participant is not None:
            return
        self._link_participant(participant)

    def _subscribe_to_tracks(self, *args) -> None:
        if self._participant is None:
            return
        if args and isinstance(args[-1], rtc.RemoteParticipant):
            # track_published: (publication, participant)
            # track_subscribed: (track, publication, participant)
            if args[-1].identity != self._participant.identity:
                return

        for publication in self._participant.track_publications.values():
            # skip tracks we don't care about
            handler = self._track_handlers.get(publication.source)
            if handler is None:
                continue

            # subscribe and setup streaming
            if not publication.subscribed:
                publication.set_subscribed(True)

            track: rtc.RemoteTrack | None = publication.track
            if track is not None:
                handler.setup(track)

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("track_published", self._subscribe_to_tracks)
        self._room.off("track_subscribed", self._subscribe_to_tracks)
        self._participant = None

        for handler in self._track_handlers.values():
            await handler.aclose()
