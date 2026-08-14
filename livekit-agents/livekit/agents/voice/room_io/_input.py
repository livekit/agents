from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, KeysView
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from typing_extensions import override

import livekit.rtc as rtc
from livekit.rtc._proto.track_pb2 import AudioTrackFeature

from ...log import logger
from ...utils import aio, log_exceptions
from ..io import AudioInput, VideoInput
from ._pre_connect_audio import PreConnectAudioHandler
from .types import NoiseCancellationParams, NoiseCancellationSelector

T = TypeVar("T", bound=rtc.AudioFrame | rtc.VideoFrame)


class _ParticipantInputStream(Generic[T], ABC):
    """
    A stream that dynamically transitions between new audio and video feeds from a connected
    participant, seamlessly switching to a different stream when the linked participant changes.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        track_source: rtc.TrackSource.ValueType | list[rtc.TrackSource.ValueType],
        processor: rtc.FrameProcessor[T] | None = None,
    ) -> None:
        self._room = room
        self._accepted_sources = (
            {track_source}
            if isinstance(track_source, rtc.TrackSource.ValueType)
            else set(track_source)
        )

        self._data_ch = aio.Chan[T]()
        self._publication: rtc.RemoteTrackPublication | None = None
        self._stream: rtc.VideoStream | rtc.AudioStream | None = None
        self._participant_identity: str | None = None
        self._attached = True

        self._forward_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

        self._room.on("track_subscribed", self._on_track_available)
        self._room.on("track_unpublished", self._on_track_unavailable)

        self._processor = processor
        self._processor_owned = False

    async def __anext__(self) -> T:
        return await self._data_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    @property
    def publication_source(self) -> rtc.TrackSource.ValueType:
        if not self._publication:
            return rtc.TrackSource.SOURCE_UNKNOWN
        return self._publication.source

    def on_attached(self) -> None:
        logger.debug(
            "input stream attached",
            extra={
                "participant": self._participant_identity,
                "source": rtc.TrackSource.Name(self.publication_source),
                "accepted_sources": [
                    rtc.TrackSource.Name(source) for source in self._accepted_sources
                ],
            },
        )
        self._attached = True

    def on_detached(self) -> None:
        logger.debug(
            "input stream detached",
            extra={
                "participant": self._participant_identity,
                "source": rtc.TrackSource.Name(self.publication_source),
                "accepted_sources": [
                    rtc.TrackSource.Name(source) for source in self._accepted_sources
                ],
            },
        )
        self._attached = False

    def set_participant(self, participant: rtc.RemoteParticipant | str | None) -> None:
        # set_participant can be called before the participant is connected
        participant_identity = (
            participant.identity if isinstance(participant, rtc.RemoteParticipant) else participant
        )
        if self._participant_identity == participant_identity:
            return

        self._participant_identity = participant_identity
        self._close_stream()

        if participant_identity is None:
            return

        participant = (
            participant
            if isinstance(participant, rtc.RemoteParticipant)
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
        self._publication = None
        if self._processor:
            self._processor._close()
            self._processor = None
        if self._forward_atask:
            await aio.cancel_and_wait(self._forward_atask)

        self._room.off("track_subscribed", self._on_track_available)
        self._room.off("track_unpublished", self._on_track_unavailable)
        self._data_ch.close()

    @log_exceptions(logger=logger)
    async def _forward_task(
        self,
        old_task: asyncio.Task[None] | None,
        stream: rtc.VideoStream | rtc.AudioStream,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
        sink: aio.Chan[T],
    ) -> None:
        if old_task:
            await aio.cancel_and_wait(old_task)

        extra = {
            "participant": participant.identity,
            "source": rtc.TrackSource.Name(publication.source),
        }
        logger.debug("start reading stream", extra=extra)
        try:
            async for event in stream:
                if not self._should_forward():
                    # drop frames if the stream is detached
                    continue
                frame = cast(T, event.frame)
                self._process_frame(frame)
                await sink.send(frame)
        except aio.ChanClosed:
            # the sink of a mixed participant is closed when they leave, mid-forwarding.
            # caught here, below @log_exceptions, so a departure isn't logged as a crash.
            logger.debug("stream sink closed", extra=extra)
            return

        logger.debug("stream closed", extra=extra)

    def _should_forward(self) -> bool:
        """Whether frames read off a participant's track are worth forwarding."""
        return self._attached

    def _process_frame(self, frame: T) -> None:
        """Hook for subclasses to process frames in-place before forwarding."""
        pass

    @abstractmethod
    def _create_stream(
        self, track: rtc.RemoteTrack, participant: rtc.Participant
    ) -> rtc.VideoStream | rtc.AudioStream: ...

    def _update_processor(self, processor: rtc.FrameProcessor[T] | None) -> None:
        if processor is None and not self._processor_owned:
            return

        old = self._processor
        if old is not None and old is not processor and self._processor_owned:
            old._close()
        self._processor = processor
        self._processor_owned = processor is not None

    def _close_stream(self) -> None:
        if self._stream is not None:
            task = asyncio.create_task(self._stream.aclose())
            task.add_done_callback(self._tasks.discard)
            self._tasks.add(task)
            self._stream = None
            self._publication = None
        self._update_processor(None)

    def _on_track_available(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> bool:
        if (
            self._participant_identity != participant.identity
            or publication.source not in self._accepted_sources
            or (self._publication and self._publication.sid == publication.sid)
        ):
            return False

        self._close_stream()
        self._stream = self._create_stream(track, participant)
        self._publication = publication
        self._forward_atask = asyncio.create_task(
            self._forward_task(
                self._forward_atask, self._stream, publication, participant, self._data_ch
            )
        )
        return True

    def _on_track_unavailable(
        self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            not self._publication
            or self._publication.sid != publication.sid
            or participant.identity != self._participant_identity
        ):
            return

        self._close_stream()

        # subscribe to the first available track
        for publication in participant.track_publications.values():
            if publication.track is None:
                continue
            if self._on_track_available(publication.track, publication, participant):
                return


@dataclass
class _MixedSource:
    """One participant's contribution to the mixed audio input."""

    chan: aio.Chan[rtc.AudioFrame]
    stream: rtc.AudioStream | None = None
    task: asyncio.Task[None] | None = None
    publication_sid: str | None = None
    processor: rtc.FrameProcessor[rtc.AudioFrame] | None = None


class _ParticipantAudioInputStream(_ParticipantInputStream[rtc.AudioFrame], AudioInput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        noise_cancellation: rtc.NoiseCancellationOptions
        | NoiseCancellationSelector
        | rtc.FrameProcessor[rtc.AudioFrame]
        | None,
        auto_gain_control: bool = True,
        pre_connect_audio_handler: PreConnectAudioHandler | None,
        frame_size_ms: int = 50,
        mix_participants: bool = False,
    ) -> None:
        _ParticipantInputStream.__init__(
            self,
            room=room,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            processor=(
                noise_cancellation if isinstance(noise_cancellation, rtc.FrameProcessor) else None
            ),
        )
        AudioInput.__init__(self, label="RoomIO")
        if frame_size_ms <= 0:
            raise ValueError("frame_size_ms must be greater than 0")

        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._frame_size_ms = frame_size_ms
        self._noise_cancellation = noise_cancellation
        self._pre_connect_audio_handler = pre_connect_audio_handler
        self._apm: rtc.AudioProcessingModule | None = None
        if auto_gain_control:
            self._apm = rtc.AudioProcessingModule(auto_gain_control=True)

        self._mix_participants = mix_participants
        self._mix_sources: dict[str, _MixedSource] = {}
        self._mixer: rtc.AudioMixer | None = None
        self._mixer_atask: asyncio.Task[None] | None = None

        if mix_participants:
            self._room.on("track_muted", self._on_track_muted)
            self._room.on("track_unmuted", self._on_track_unmuted)

        if mix_participants and isinstance(noise_cancellation, rtc.FrameProcessor):
            logger.warning(
                "a single noise cancellation processor is shared by every mixed participant, "
                "which interleaves speakers through one stateful filter. pass a callable "
                "returning a new processor per participant instead",
                extra={"processor": type(noise_cancellation).__name__},
            )

    @property
    def mix_participants(self) -> bool:
        return self._mix_participants

    @property
    def mixed_identities(self) -> KeysView[str]:
        return self._mix_sources.keys()

    def add_participant(self, participant: rtc.RemoteParticipant) -> None:
        """Mix this participant's microphone into the input stream.

        Only used when `mix_participants` is enabled; a no-op otherwise.
        """
        if not self._mix_participants or participant.identity in self._mix_sources:
            return

        source = _MixedSource(chan=aio.Chan[rtc.AudioFrame]())
        self._mix_sources[participant.identity] = source
        self._ensure_mixer()

        for publication in participant.track_publications.values():
            if publication.track and self._on_track_available(
                publication.track, publication, participant
            ):
                break

    def remove_participant(self, identity: str) -> None:
        source = self._mix_sources.pop(identity, None)
        if source is None:
            return

        self._close_mixed_source(source)
        source.chan.close()

    def _ensure_mixer(self) -> rtc.AudioMixer:
        if self._mixer is None:
            self._mixer = rtc.AudioMixer(
                self._sample_rate,
                self._num_channels,
                blocksize=int(self._sample_rate * self._frame_size_ms / 1000),
                # must stay above the frame interval: the mixer pads a stream that misses this
                # deadline, so a live stream would get chopped by a tighter timeout
                stream_timeout_ms=max(100, self._frame_size_ms * 2),
            )
            self._mixer_atask = asyncio.create_task(self._forward_mixed(self._mixer))
        return self._mixer

    def _on_track_muted(
        self, participant: rtc.Participant, publication: rtc.TrackPublication
    ) -> None:
        source = self._mix_sources.get(participant.identity)
        if source is None or source.publication_sid != publication.sid:
            return

        # a muted track delivers nothing: stop reading it until it comes back
        self._close_mixed_source(source)

    def _on_track_unmuted(
        self, participant: rtc.Participant, publication: rtc.TrackPublication
    ) -> None:
        if self._mix_sources.get(participant.identity) is None or not publication.track:
            return

        self._on_track_available(publication.track, publication, participant)  # type: ignore[arg-type]

    def _close_mixed_source(self, source: _MixedSource) -> None:
        stream, task = source.stream, source.task
        source.stream, source.task, source.publication_sid = None, None, None

        # the mixer waits stream_timeout_ms on every registered stream for every block, so a
        # source that stopped delivering paces the whole mix below real time
        if self._mixer is not None:
            self._mixer.remove_stream(source.chan)
        while not source.chan.empty():
            source.chan.recv_nowait()  # in-flight audio would resurface on the next unmute

        async def _close() -> None:
            if task:
                await aio.cancel_and_wait(task)
            if stream:
                await stream.aclose()

        close_task = asyncio.create_task(_close())
        close_task.add_done_callback(self._tasks.discard)
        self._tasks.add(close_task)

    @log_exceptions(logger=logger)
    async def _forward_mixed(self, mixer: rtc.AudioMixer) -> None:
        async for frame in mixer:
            if not self._attached:
                continue
            if self._apm is not None:
                self._apm.process_stream(frame)
            await self._data_ch.send(frame)

    @override
    def set_participant(self, participant: rtc.RemoteParticipant | str | None) -> None:
        if self._mix_participants:
            # every accepted participant is mixed in, linking only drives the outputs
            return
        super().set_participant(participant)

    @override
    def _on_track_available(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> bool:
        if not self._mix_participants:
            return super()._on_track_available(track, publication, participant)

        source = self._mix_sources.get(participant.identity)
        if (
            source is None
            or publication.source not in self._accepted_sources
            or source.publication_sid == publication.sid
        ):
            return False

        self._close_mixed_source(source)
        if publication.muted:
            return False  # nothing to read yet, track_unmuted brings the source back

        source.publication_sid = publication.sid
        source.stream = self._create_stream(track, participant)
        # the sink is bound here, once: a participant removed mid-forward must never fall
        # back to _data_ch and reach the session unmixed
        source.task = asyncio.create_task(
            self._forward_task(
                None, source.stream, publication, participant, source.chan, source.processor
            )
        )
        self._ensure_mixer().add_stream(source.chan)
        return True

    @override
    def _on_track_unavailable(
        self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if not self._mix_participants:
            super()._on_track_unavailable(publication, participant)
            return

        source = self._mix_sources.get(participant.identity)
        if source is None or source.publication_sid != publication.sid:
            return

        self._close_mixed_source(source)
        for publication in participant.track_publications.values():
            if publication.track and self._on_track_available(
                publication.track, publication, participant
            ):
                return

    @override
    def _should_forward(self) -> bool:
        # a mixed source keeps feeding the mixer while detached and _forward_mixed drops the
        # output instead: starving the mixer would pace it below real time for everyone else
        return self._mix_participants or self._attached

    @override
    def _process_frame(self, frame: rtc.AudioFrame) -> None:
        if self._mix_participants:
            return  # AGC runs once on the mixed output instead

        if self._apm is not None:
            self._apm.process_stream(frame)

    @override
    def _create_stream(self, track: rtc.Track, participant: rtc.Participant) -> rtc.AudioStream:
        noise_cancellation = self._noise_cancellation
        auto_close_noise_cancellation = False
        if callable(noise_cancellation):
            noise_cancellation = noise_cancellation(NoiseCancellationParams(participant, track))
            if self._mix_participants:
                # each mixed participant gets its own processor, tied to its stream. it is kept
                # on the source so this participant's pre-connect buffer runs through it too
                auto_close_noise_cancellation = isinstance(noise_cancellation, rtc.FrameProcessor)
                if (source := self._mix_sources.get(participant.identity)) is not None:
                    source.processor = (
                        noise_cancellation
                        if isinstance(noise_cancellation, rtc.FrameProcessor)
                        else None
                    )
            elif isinstance(noise_cancellation, rtc.FrameProcessor):
                self._update_processor(noise_cancellation)
            else:
                self._update_processor(None)

        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            frame_size_ms=self._frame_size_ms,
            noise_cancellation=noise_cancellation,
            auto_close_noise_cancellation=auto_close_noise_cancellation,
        )

    @override
    async def _forward_task(  # type: ignore[override]
        self,
        old_task: asyncio.Task[None] | None,
        stream: rtc.AudioStream,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
        sink: aio.Chan[rtc.AudioFrame],
        processor: rtc.FrameProcessor[rtc.AudioFrame] | None = None,
    ) -> None:
        """`processor` filters the pre-connect buffer; the live track is filtered by the stream
        itself. Defaults to the stream-wide processor, mixed sources pass their own."""
        if old_task:
            await aio.cancel_and_wait(old_task)

        if processor is None:
            processor = self._processor

        # a mixed participant's sink is closed when they leave, mid-forwarding. the base loop
        # handles that itself; this covers the pre-connect and flush sends around it.
        with contextlib.suppress(aio.ChanClosed):
            if (
                self._pre_connect_audio_handler
                and publication.track
                and AudioTrackFeature.TF_PRECONNECT_BUFFER in publication.audio_features
            ):
                logging_extra = {
                    "track_id": publication.track.sid,
                    "participant": participant.identity,
                }
                try:
                    duration: float = 0
                    frames = await self._pre_connect_audio_handler.wait_for_data(
                        publication.track.sid
                    )
                    for frame in self._resample_frames(
                        self._apply_audio_processor(frames, processor)
                    ):
                        if self._should_forward():
                            await sink.send(frame)
                            duration += frame.duration
                    if frames:
                        logger.debug(
                            "pre-connect audio buffer pushed",
                            extra={"duration": duration, **logging_extra},
                        )

                except asyncio.TimeoutError:
                    logger.warning(
                        "timeout waiting for pre-connect audio buffer",
                        extra=logging_extra,
                    )

                except Exception as e:
                    logger.error(
                        "error reading pre-connect audio buffer", extra=logging_extra, exc_info=e
                    )

            await super()._forward_task(old_task, stream, publication, participant, sink)

            # push a silent frame to flush the stt final result if any
            silent_samples = int(self._sample_rate * 0.5)
            await sink.send(
                rtc.AudioFrame(
                    b"\x00\x00" * silent_samples,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=silent_samples,
                )
            )

    @override
    async def aclose(self) -> None:
        if self._mix_participants:
            self._room.off("track_muted", self._on_track_muted)
            self._room.off("track_unmuted", self._on_track_unmuted)

        sources, self._mix_sources = list(self._mix_sources.values()), {}
        for source in sources:
            source.chan.close()
            if source.task:
                await aio.cancel_and_wait(source.task)
            if source.stream:
                await source.stream.aclose()

        if self._mixer_atask:
            await aio.cancel_and_wait(self._mixer_atask)
            self._mixer_atask = None
        if self._mixer:
            await self._mixer.aclose()
            self._mixer = None

        await super().aclose()

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

    def _apply_audio_processor(
        self,
        frames: Iterable[rtc.AudioFrame],
        processor: rtc.FrameProcessor[rtc.AudioFrame] | None,
    ) -> Iterable[rtc.AudioFrame]:
        for frame in frames:
            if processor is not None:
                try:
                    yield processor._process(frame)
                except Exception as e:
                    logger.warning(
                        "error pre-processing audio frame: %s",
                        e,
                    )
                    yield frame
            else:
                yield frame


class _ParticipantVideoInputStream(_ParticipantInputStream[rtc.VideoFrame], VideoInput):
    def __init__(self, room: rtc.Room) -> None:
        _ParticipantInputStream.__init__(
            self,
            room=room,
            track_source=[
                rtc.TrackSource.SOURCE_CAMERA,
                rtc.TrackSource.SOURCE_SCREENSHARE,
            ],
        )
        VideoInput.__init__(self, label="RoomIO")

    @override
    def _create_stream(self, track: rtc.Track, participant: rtc.Participant) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track)
