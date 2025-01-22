from __future__ import annotations

import asyncio
from typing import Optional

from livekit import rtc

from ..log import logger
from ..utils import aio
from .io import AudioStream, VideoStream


class RoomInput:
    """Creates video and audio streams from a remote participant in a LiveKit room"""

    class _RemoteTrackStreamer:
        """Manages streaming from a remote track to a channel"""

        def __init__(
            self,
            source: rtc.TrackSource.ValueType,
            enabled: bool = False,
            sample_rate: int | None = None,
        ) -> None:
            self.source = source
            self.enabled = enabled
            self.sample_rate = sample_rate

            self.track: rtc.RemoteTrack | None = None
            self.task: asyncio.Task | None = None
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

            if self.task is not None:
                self.task.cancel()

            assert self._data_ch is not None
            self.track = track
            stream = (
                rtc.AudioStream(track, sample_rate=self.sample_rate)
                if self.source == rtc.TrackSource.SOURCE_MICROPHONE
                else rtc.VideoStream(track)
            )
            self.task = asyncio.create_task(self._stream_frames(stream))

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
    ) -> None:
        """
        Args:
            room: The LiveKit room to get streams from
            participant_identity: Optional identity of the participant to get streams from.
                                If None, will use the first participant that joins.
            audio_enabled: Whether to enable audio input
            video_enabled: Whether to enable video input
        """
        self._room = room
        self._expected_identity = participant_identity
        self._participant: rtc.RemoteParticipant | None = None
        self._closed = False

        # set up track streamers
        self._audio_streamer = self._RemoteTrackStreamer(
            rtc.TrackSource.SOURCE_MICROPHONE, enabled=audio_enabled, sample_rate=16000
        )
        self._video_streamer = self._RemoteTrackStreamer(
            rtc.TrackSource.SOURCE_CAMERA, enabled=video_enabled
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

    def _subscribe_to_tracks(self, *args, **kwargs) -> None:
        if self._participant is None:
            return

        for publication in self._participant.track_publications.values():
            # skip tracks we don't care about
            streamer = None
            if publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                streamer = self._audio_streamer
            elif publication.source == rtc.TrackSource.SOURCE_CAMERA:
                streamer = self._video_streamer

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
            if streamer.task is not None:
                await aio.graceful_cancel(streamer.task)
