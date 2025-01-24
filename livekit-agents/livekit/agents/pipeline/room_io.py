import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from livekit import rtc

from .io import AudioSink


# TODO: add RoomOutput that has audio and video sinks, optionally with av sync?
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
    audio_enabled: bool = True
    """Whether to subscribe to audio"""
    video_enabled: bool = False
    """Whether to subscribe to video"""
    audio_sample_rate: int = 16000
    """Sample rate of the input audio in Hz"""
    audio_num_channels: int = 1
    """Number of audio channels"""
    audio_queue_capacity: int = 0
    """Capacity of the internal audio queue, 0 means unlimited"""
    video_queue_capacity: int = 0
    """Capacity of the internal video queue, 0 means unlimited"""


DEFAULT_ROOM_INPUT_OPTIONS = RoomInputOptions()


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

        # streams
        self._audio_stream: Optional[rtc.AudioStream] = None
        self._video_stream: Optional[rtc.VideoStream] = None

        self._participant_ready = asyncio.Event()
        self._room.on("participant_connected", self._on_participant_connected)

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
    def audio(self) -> AsyncIterator[rtc.AudioFrame] | None:
        if self._audio_stream is None:
            return None

        async def _read_stream():
            async for event in self._audio_stream:
                yield event.frame

        return _read_stream()

    @property
    def video(self) -> AsyncIterator[rtc.VideoFrame] | None:
        if self._video_stream is None:
            return None

        async def _read_stream():
            async for event in self._video_stream:
                yield event.frame

        return _read_stream()

    def _link_participant(self, participant: rtc.RemoteParticipant) -> None:
        if (
            self._expected_identity is not None
            and participant.identity != self._expected_identity
        ):
            return

        self._participant = participant

        # set up tracks
        if self._options.audio_enabled:
            self._audio_stream = rtc.AudioStream.from_participant(
                participant=participant,
                track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                sample_rate=self._options.audio_sample_rate,
                num_channels=self._options.audio_num_channels,
                capacity=self._options.audio_queue_capacity,
            )
        if self._options.video_enabled:
            self._video_stream = rtc.VideoStream.from_participant(
                participant=participant,
                track_source=rtc.TrackSource.SOURCE_CAMERA,
                capacity=self._options.video_queue_capacity,
            )

        self._participant_ready.set()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        if self._participant is not None:
            return
        self._link_participant(participant)

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._participant = None

        if self._audio_stream is not None:
            await self._audio_stream.aclose()
            self._audio_stream = None
        if self._video_stream is not None:
            await self._video_stream.aclose()
            self._video_stream = None
