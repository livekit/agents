from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Generic, TypeVar, Union

from typing_extensions import override

import livekit.rtc as rtc
from livekit.agents import utils

from ...log import logger
from ..io import AudioInput, VideoInput
from ._pre_connect_audio import PreConnectAudioData

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
            if not self._attached:
                # drop frames if the stream is detached
                continue
            await self._data_ch.send(event.frame)

        logger.debug("stream closed", extra=extra)

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
        pre_connect_audio: PreConnectAudioData | None,
        pre_connect_audio_timeout: float,
    ) -> None:
        _ParticipantInputStream.__init__(
            self, room=room, track_source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._noise_cancellation = noise_cancellation
        self._pre_connect_audio = pre_connect_audio
        self._pre_connect_audio_timeout = pre_connect_audio_timeout

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.AudioStream:
        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            noise_cancellation=self._noise_cancellation,
        )

    @override
    async def _forward_task(self, old_task: asyncio.Task | None, stream: rtc.AudioStream) -> None:
        if self._pre_connect_audio:
            try:
                duration = 0
                frames = await self._pre_connect_audio.wait_for_data(
                    timeout=self._pre_connect_audio_timeout
                )
                for frame in self._resample_frames(frames):
                    if self._attached:
                        await self._data_ch.send(frame)
                        duration += frame.duration
                if duration > 0:
                    logger.debug("pre-connect audio buffer pushed", extra={"duration": duration})

            except asyncio.TimeoutError:
                logger.warning("timeout waiting for pre-connect audio buffer")

            except Exception as e:
                logger.error("error reading pre-connect audio buffer", extra={"error": e})

        await super()._forward_task(old_task, stream)

    def _resample_frames(self, frames: Iterable[rtc.AudioFrame]) -> Iterable[rtc.AudioFrame]:
        resampler: rtc.AudioResampler | None = None
        for frame in frames:
            if (
                not resampler
                and self._sample_rate is not None
                and frame.sample_rate != self._sample_rate
            ):
                resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate, output_rate=self._sample_rate
                )

            if resampler:
                yield from resampler.push(frame)
            else:
                yield frame

        if resampler:
            yield from resampler.flush()


class _ParticipantVideoInputStream(_ParticipantInputStream[rtc.VideoFrame], VideoInput):
    def __init__(self, room: rtc.Room) -> None:
        _ParticipantInputStream.__init__(
            self, room=room, track_source=rtc.TrackSource.SOURCE_CAMERA
        )

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track)
