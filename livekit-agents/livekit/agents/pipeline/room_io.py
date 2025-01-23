from __future__ import annotations

import asyncio
from functools import partial
from typing import Callable, Optional

from livekit import rtc

from ..utils import aio
from .io import AudioSink, AudioStream, VideoStream


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
            # Notify that playback finished
            self.on_playback_finished(
                playback_position=self._pushed_duration, interrupted=False
            )
            self._pushed_duration = None

    def clear_buffer(self) -> None:
        """Clear the audio buffer immediately"""
        # Clear the buffer
        self._audio_source.clear_queue()

        if self._pushed_duration is not None:
            # Notify that playback was interrupted
            self.on_playback_finished(
                playback_position=self._pushed_duration, interrupted=True
            )
            self._pushed_duration = None


class RoomInput:
    """Creates video and audio streams from a remote participant in a LiveKit room"""

    class _RemoteTrackHandler:
        """Manages streaming from a remote track to a aio.Chan"""

        def __init__(
            self,
            track_to_stream: Callable[
                [rtc.RemoteTrack], rtc.AudioStream | rtc.VideoStream
            ],
            enabled: bool = False,
        ) -> None:
            self.track_to_stream = track_to_stream
            self.enabled = enabled

            self.track: rtc.RemoteTrack | None = None
            self.stream_task: asyncio.Task | None = None
            self._data_ch: aio.Chan[rtc.AudioFrame | rtc.VideoFrame] | None = None

            if enabled:
                self._data_ch = aio.Chan()

        @property
        def data_ch(self) -> aio.Chan | None:
            return self._data_ch

        def setup(self, track: rtc.RemoteTrack) -> None:
            """Set up streaming for a new track"""
            if track == self.track:
                return

            if self.stream_task is not None:
                self.stream_task.cancel()

            assert self._data_ch is not None
            self.track = track
            stream = self.track_to_stream(track)
            self.stream_task = asyncio.create_task(self._stream_frames(stream))

        async def _stream_frames(
            self, stream: rtc.AudioStream | rtc.VideoStream
        ) -> None:
            assert self._data_ch is not None
            async for event in stream:
                self._data_ch.send_nowait(event.frame)

    def __init__(
        self,
        room: rtc.Room,
        participant_identity: Optional[str] = None,
        *,
        audio_enabled: bool = True,
        video_enabled: bool = False,
        audio_sample_rate: int = 16000,
        audio_num_channels: int = 1,
    ) -> None:
        """
        Args:
            room: The LiveKit room to get streams from
            participant_identity: Optional identity of the participant to get streams from.
                                If None, will use the first participant that joins.
            audio_enabled: Whether to enable audio input
            video_enabled: Whether to enable video input
            audio_sample_rate: Sample rate of the audio in Hz
            audio_num_channels: Number of audio channels
        """
        self._room = room
        self._expected_identity = participant_identity
        self._participant: rtc.RemoteParticipant | None = None
        self._closed = False

        # set up track streamers
        self._audio_streamer = self._RemoteTrackHandler(
            track_to_stream=partial(
                rtc.AudioStream,
                sample_rate=audio_sample_rate,
                num_channels=audio_num_channels,
            ),
            enabled=audio_enabled,
        )
        self._video_streamer = self._RemoteTrackHandler(
            track_to_stream=rtc.VideoStream,
            enabled=video_enabled,
        )

        self._participant_ready = asyncio.Event()
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("track_published", self._subscribe_to_tracks)
        self._room.on("track_subscribed", self._subscribe_to_tracks)

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
        return self._audio_streamer.data_ch

    @property
    def video(self) -> VideoStream | None:
        return self._video_streamer.data_ch

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
            streamer = {
                rtc.TrackSource.SOURCE_MICROPHONE: self._audio_streamer,
                rtc.TrackSource.SOURCE_CAMERA: self._video_streamer,
            }.get(publication.source)

            if streamer is None or not streamer.enabled:
                continue

            # subscribe and setup streaming
            if not publication.subscribed:
                publication.set_subscribed(True)

            track: rtc.RemoteTrack | None = publication.track
            if track is not None:
                streamer.setup(track)

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("RoomInput already closed")

        self._closed = True
        self._room.off("participant_connected", self._on_participant_connected)
        self._room.off("track_published", self._subscribe_to_tracks)
        self._room.off("track_subscribed", self._subscribe_to_tracks)
        self._participant = None

        # Cancel stream tasks
        for streamer in [self._audio_streamer, self._video_streamer]:
            if streamer.stream_task is not None:
                await aio.graceful_cancel(streamer.stream_task)
