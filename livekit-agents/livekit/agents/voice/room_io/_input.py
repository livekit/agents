from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar, Union

from typing_extensions import override

import livekit.rtc as rtc
from livekit.agents import utils

from ...log import logger
from ..io import AudioInput, VideoInput

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
        self._data_ch = utils.aio.Chan[T]()
        self._stream: rtc.VideoStream | rtc.AudioStream | None = None
        self._participant_identity: str | None = None
        self._attached = True

        self._forward_atask: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

        self._room.on("track_subscribed", self._on_track_available)

    async def __anext__(self) -> T:
        return await self._data_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    def on_attached(self) -> None:
        logger.debug(
            "input stream attached",
            extra={
                "participant": self._participant_identity,
                "source": rtc.TrackSource.Name(self._track_source),
            },
        )
        self._attached = True

    def on_detached(self) -> None:
        logger.debug(
            "input stream detached",
            extra={
                "participant": self._participant_identity,
                "source": rtc.TrackSource.Name(self._track_source),
            },
        )
        self._attached = False

    def set_participant(self, participant: rtc.Participant | str | None) -> None:
        # set_participant can be called before the participant is connected
        participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )
        if self._participant_identity == participant_identity:
            return

        self._participant_identity = participant_identity

        if participant_identity is None:
            self._close_stream()
            return

        participant = (
            participant
            if isinstance(participant, rtc.Participant)
            else self._room.remote_participants.get(participant_identity)
        )
        if participant:
            for publication in participant.track_publications.values():
                if not publication.track:
                    continue
                self._on_track_available(publication.track, publication, participant)

    async def aclose(self) -> None:
        if self._stream:
            await self._stream.aclose()
            self._stream = None
        if self._forward_atask:
            await utils.aio.cancel_and_wait(self._forward_atask)

        self._room.off("track_subscribed", self._on_track_available)
        self._data_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _forward_task(
        self, old_task: asyncio.Task | None, stream: rtc.VideoStream | rtc.AudioStream
    ) -> None:
        if old_task:
            await utils.aio.cancel_and_wait(old_task)

        extra = {
            "participant": self._participant_identity,
            "source": rtc.TrackSource.Name(self._track_source),
        }
        logger.debug("start reading stream", extra=extra)
        async for event in stream:
            await self._push_data(event)

        logger.debug("stream closed", extra=extra)

    async def _push_data(self, event: rtc.AudioFrameEvent | rtc.VideoFrameEvent) -> None:
        if not self._attached:
            # drop frames if the stream is detached
            return

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
        if (
            self._participant_identity != participant.identity
            or publication.source != self._track_source
        ):
            return

        self._close_stream()
        self._stream = self._create_stream(track)
        self._forward_atask = asyncio.create_task(
            self._forward_task(self._forward_atask, self._stream)
        )


class _ParticipantAudioInputStream(_ParticipantInputStream[rtc.AudioFrame], AudioInput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        noise_cancellation: rtc.NoiseCancellationOptions | None,
        pre_attach_buffer_ms: int = 0,
    ) -> None:
        _ParticipantInputStream.__init__(
            self, room=room, track_source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._noise_cancellation = noise_cancellation
        self._pre_attach_buffer_ms = pre_attach_buffer_ms

        self._pre_attach_buffer: utils.audio.AudioRingBuffer | None = None

    @override
    def on_detached(self) -> None:
        super().on_detached()
        if self._pre_attach_buffer_ms > 0:
            self._pre_attach_buffer = utils.audio.AudioRingBuffer(
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                buffer_ms=self._pre_attach_buffer_ms,
            )

    @override
    async def _push_data(self, event: rtc.AudioFrameEvent) -> None:
        if self._pre_attach_buffer:
            self._pre_attach_buffer.push(event.frame.data)

        if not self._attached:
            return
        elif self._pre_attach_buffer:
            # push pre-attach buffer frames
            frames = self._pre_attach_buffer.flush()
            for frame in frames:
                self._data_ch.send_nowait(frame)
            self._pre_attach_buffer = None

        await self._data_ch.send(event.frame)

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.AudioStream:
        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            noise_cancellation=self._noise_cancellation,
        )


class _ParticipantVideoInputStream(_ParticipantInputStream[rtc.VideoFrame], VideoInput):
    def __init__(self, room: rtc.Room) -> None:
        _ParticipantInputStream.__init__(
            self, room=room, track_source=rtc.TrackSource.SOURCE_CAMERA
        )

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track)
