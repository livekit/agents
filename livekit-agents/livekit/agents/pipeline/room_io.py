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


@dataclass
class _TrackState:
    """State for a single audio or video track"""

    stream: aio.Chan[rtc.AudioFrame | rtc.VideoFrame]
    enabled: bool = True
    task: Optional[asyncio.Task] = None
    track: Optional[rtc.RemoteTrack] = None


class RoomInput:
    """Creates video and audio streams from a remote participant in a LiveKit room"""

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

        # Track state
        self._track_states: dict[rtc.TrackSource.ValueType, _TrackState] = {}

        # Initialize track state for subscribed sources
        if self._options.subscribe_audio:
            source = rtc.TrackSource.SOURCE_MICROPHONE
            self._track_states[source] = _TrackState(stream=aio.Chan())

        if self._options.subscribe_video:
            source = rtc.TrackSource.SOURCE_CAMERA
            self._track_states[source] = _TrackState(stream=aio.Chan())

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
        if rtc.TrackSource.SOURCE_MICROPHONE not in self._track_states:
            return None
        return self._track_states[rtc.TrackSource.SOURCE_MICROPHONE].stream

    @property
    def video(self) -> VideoStream | None:
        if rtc.TrackSource.SOURCE_CAMERA not in self._track_states:
            return None
        return self._track_states[rtc.TrackSource.SOURCE_CAMERA].stream

    @property
    def audio_enabled(self) -> bool:
        if rtc.TrackSource.SOURCE_MICROPHONE not in self._track_states:
            return False
        return self._track_states[rtc.TrackSource.SOURCE_MICROPHONE].enabled

    @property
    def video_enabled(self) -> bool:
        if rtc.TrackSource.SOURCE_CAMERA not in self._track_states:
            return False
        return self._track_states[rtc.TrackSource.SOURCE_CAMERA].enabled

    def set_enabled(self, source: rtc.TrackSource.ValueType, enabled: bool) -> None:
        if source not in self._track_states:
            raise ValueError(f"Track {source} is not subscribed")
        self._track_states[source].enabled = enabled

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
            if publication.source not in self._track_states:
                continue

            # subscribe and setup streaming
            if not publication.subscribed:
                publication.set_subscribed(True)

            track: rtc.RemoteTrack | None = publication.track
            if track is not None:
                self._setup_track(publication.source, track)

    def _setup_track(
        self, source: rtc.TrackSource.ValueType, track: rtc.RemoteTrack
    ) -> None:
        """Set up streaming for a new track"""
        state = self._track_states[source]
        if track == state.track:
            return

        # Cancel existing task if any
        if state.task is not None:
            state.task.cancel()

        state.track = track
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

        async def _read_stream(stream: rtc.AudioStream | rtc.VideoStream) -> None:
            """Handle reading from an audio or video stream"""

            async for event in stream:
                if not state.enabled:
                    continue
                state.stream.send_nowait(event.frame)
                if (
                    isinstance(event.frame, rtc.VideoFrame)
                    and self._options.video_buffer_size is not None
                ):
                    dropped = 0
                    while state.stream.qsize() > self._options.video_buffer_size:
                        await state.stream.recv()  # drop old frames if buffer is full
                        dropped += 1
                    if dropped > 0 and self._options.warn_dropped_video_frames:
                        logger.warning(
                            "dropping video frames since buffer is full",
                            extra={
                                "buffer_size": self._options.video_buffer_size,
                                "dropped": dropped,
                            },
                        )

        state.task = asyncio.create_task(_read_stream(stream))

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("track_published", self._subscribe_to_tracks)
        self._room.off("track_subscribed", self._subscribe_to_tracks)
        self._participant = None

        # Cancel all track tasks and close channels
        for state in self._track_states.values():
            if state.task is not None:
                await aio.graceful_cancel(state.task)
            state.stream.close()
        self._track_states.clear()
