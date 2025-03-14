from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterable, Generic, TypeVar, Union

import livekit.rtc as rtc
from livekit.agents import utils
from typing_extensions import override

from ...log import logger

T = TypeVar("T", bound=Union[rtc.AudioFrame, rtc.VideoFrame])


class _ParticipantInputStream(Generic[T], ABC):
    """
    A stream that dynamically transitions between new audio and video feeds from a connected
    participant, seamlessly switching to a different stream when the linked participant changes.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        track_source: rtc.TrackSource.ValueType,
    ) -> None:
        self._room, self._track_source = room, track_source
        self._data_ch = utils.aio.Chan()
        self._stream: rtc.VideoStream | rtc.AudioStream | None = None
        self._participant: rtc.RemoteParticipant | None = None
        self._forward_atask: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._room.on("track_subscribed", self._on_track_available)

    @property
    def stream(self) -> AsyncIterable[T]:
        return self._data_ch

    def set_participant(self, participant: rtc.RemoteParticipant | None) -> None:
        self._participant = participant

        if participant is None:
            self._close_stream()
            return

        for publication in participant.track_publications.values():
            if not publication.track:
                continue

            self._on_track_available(publication.track, publication, participant)

    async def aclose(self) -> None:
        if self._forward_atask:
            await utils.aio.cancel_and_wait(self._forward_atask)

        self._room.off("track_subscribed", self._on_track_available)
        self._close_stream()
        self._data_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _forward_task(
        self, old_task: asyncio.Task | None, stream: rtc.VideoStream | rtc.AudioStream
    ) -> None:
        if old_task:
            await utils.aio.cancel_and_wait(old_task)

        async for event in stream:
            await self._data_ch.send(event.frame)

    @abstractmethod
    def _create_stream(self, track: rtc.RemoteTrack) -> rtc.VideoStream | rtc.AudioStream: ...

    def _close_stream(self) -> None:
        if self._stream is not None:
            task = asyncio.create_task(self._stream.aclose())
            task.add_done_callback(self._tasks.discard)
            self._tasks.add(task)
            self._stream = None

    def _on_track_available(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if self._participant != participant or publication.source != self._track_source:
            return

        self._close_stream()
        self._stream = self._create_stream(track)
        self._forward_atask = asyncio.create_task(
            self._forward_task(self._forward_atask, self._stream)
        )


class _ParticipantAudioInputStream(_ParticipantInputStream[rtc.AudioFrame]):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        noise_cancellation: rtc.NoiseCancellationOptions | None,
    ) -> None:
        super().__init__(room=room, track_source=rtc.TrackSource.SOURCE_MICROPHONE)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._noise_cancellation = noise_cancellation

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.AudioStream:
        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            noise_cancellation=self._noise_cancellation,
        )


class _ParticipantVideoInputStream(_ParticipantInputStream[rtc.VideoFrame]):
    def __init__(self, room: rtc.Room) -> None:
        super().__init__(room=room, track_source=rtc.TrackSource.SOURCE_CAMERA)

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track)
