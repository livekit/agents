from __future__ import annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Generic, TypeVar, cast

from typing_extensions import override

import livekit.rtc as rtc
from livekit.rtc._proto.track_pb2 import AudioTrackFeature

from ...log import logger
from ...utils import aio, audio, log_exceptions
from ..io import AudioInput, VideoInput
from ._pre_connect_audio import PreConnectAudioHandler
from .types import NoiseCancellationParams, NoiseCancellationSelector, ParticipantsMode

T = TypeVar("T", bound=rtc.AudioFrame | rtc.VideoFrame)

_TURN_GAP = 0.5
"""Silence pushed to the session to close a turn, so the STT flushes what it has."""
_PREROLL = 2.0
"""Audio held per participant before they are selected as the active speaker."""


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
        on_stream_changed: Callable[[], None] | None = None,
    ) -> None:
        self._room = room
        self._on_stream_changed = on_stream_changed
        self._accepted_sources = (
            {track_source}
            if isinstance(track_source, rtc.TrackSource.ValueType)
            else set(track_source)
        )

        self._data_ch = aio.Chan[T]()
        self._publication: rtc.RemoteTrackPublication | None = None
        self._stream: rtc.VideoStream | rtc.AudioStream | None = None
        self._streaming = False
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

    @property
    def streaming(self) -> bool:
        """Whether a live, unmuted track is currently being read."""
        return self._streaming and not (self._publication and self._publication.muted)

    def _set_streaming(self, streaming: bool) -> None:
        if streaming == self._streaming:
            return

        self._streaming = streaming
        if self._on_stream_changed:
            self._on_stream_changed()

    def _drain(self) -> None:
        """Drop whatever is buffered but not consumed yet."""
        while not self._data_ch.empty():
            self._data_ch.recv_nowait()

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
    ) -> None:
        if old_task:
            await aio.cancel_and_wait(old_task)

        extra = {
            "participant": participant.identity,
            "source": rtc.TrackSource.Name(publication.source),
        }
        logger.debug("start reading stream", extra=extra)
        async for event in stream:
            if not self._attached:
                # drop frames if the stream is detached
                continue
            frame = cast(T, event.frame)
            self._process_frame(frame)
            await self._data_ch.send(frame)

        # a track can stop delivering without being unpublished, e.g. the publisher closing it
        self._set_streaming(False)
        logger.debug("stream closed", extra=extra)

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
            self._set_streaming(False)
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
            self._forward_task(self._forward_atask, self._stream, publication, participant)
        )
        self._set_streaming(True)
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


class _RoomAudioInput(AudioInput, ABC):
    """The audio input `RoomIO` drives.

    `RoomIO` owns the participant lifecycle and routes it here: `add_participant` and
    `remove_participant` are called for every accepted participant, `set_participant` for
    the linked one. An input that only listens to the linked participant ignores the former.
    """

    @abstractmethod
    def set_participant(self, participant: rtc.RemoteParticipant | str | None) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    def add_participant(self, participant: rtc.RemoteParticipant) -> None:
        """Called for every accepted participant.

        Does nothing by default: an input that only listens to the linked participant has
        nothing to do with the others, and `set_participant` already tells it who that is.
        """
        pass

    def remove_participant(self, identity: str) -> None:
        """Called when a participant leaves. Does nothing by default, see `add_participant`."""
        pass


class _ParticipantAudioInputStream(_ParticipantInputStream[rtc.AudioFrame], _RoomAudioInput):
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
        on_stream_changed: Callable[[], None] | None = None,
    ) -> None:
        _ParticipantInputStream.__init__(
            self,
            room=room,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            processor=(
                noise_cancellation if isinstance(noise_cancellation, rtc.FrameProcessor) else None
            ),
            on_stream_changed=on_stream_changed,
        )
        _RoomAudioInput.__init__(self, label="RoomIO")
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

    @override
    def _process_frame(self, frame: rtc.AudioFrame) -> None:
        if self._apm is not None:
            self._apm.process_stream(frame)

    @override
    def _create_stream(self, track: rtc.Track, participant: rtc.Participant) -> rtc.AudioStream:
        noise_cancellation = self._noise_cancellation
        if callable(noise_cancellation):
            noise_cancellation = noise_cancellation(NoiseCancellationParams(participant, track))
            if isinstance(noise_cancellation, rtc.FrameProcessor):
                self._update_processor(noise_cancellation)
            else:
                self._update_processor(None)

        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            frame_size_ms=self._frame_size_ms,
            noise_cancellation=noise_cancellation,
            auto_close_noise_cancellation=False,
        )

    @override
    async def _forward_task(
        self,
        old_task: asyncio.Task[None] | None,
        stream: rtc.AudioStream,  # type: ignore[override]
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if old_task:
            await aio.cancel_and_wait(old_task)

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
                frames = await self._pre_connect_audio_handler.wait_for_data(publication.track.sid)
                for frame in self._resample_frames(self._apply_audio_processor(frames)):
                    if self._attached:
                        await self._data_ch.send(frame)
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

        await super()._forward_task(old_task, stream, publication, participant)

        # push a silent frame to flush the stt final result if any
        await self._data_ch.send(
            audio.silence_frame(_TURN_GAP, self._sample_rate, self._num_channels)
        )

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

    def _apply_audio_processor(self, frames: Iterable[rtc.AudioFrame]) -> Iterable[rtc.AudioFrame]:
        for frame in frames:
            if self._processor is not None:
                try:
                    yield self._processor._process(frame)
                except Exception as e:
                    logger.warning(
                        "error pre-processing audio frame: %s",
                        e,
                    )
                    yield frame
            else:
                yield frame


class _ParticipantAudioInputGroup(_RoomAudioInput, ABC):
    """Several participants behind a single audio input.

    Owns one unchanged :class:`_ParticipantAudioInputStream` per participant; subclasses
    decide how those are combined into the frames the session reads.
    """

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
    ) -> None:
        super().__init__(label="RoomIO")
        self._room = room
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._frame_size_ms = frame_size_ms
        self._attached = True
        self._streams: dict[str, _ParticipantAudioInputStream] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._data_ch = aio.Chan[rtc.AudioFrame]()

        self._new_stream = functools.partial(
            _ParticipantAudioInputStream,
            room,
            sample_rate=sample_rate,
            num_channels=num_channels,
            noise_cancellation=noise_cancellation,
            auto_gain_control=auto_gain_control,
            pre_connect_audio_handler=pre_connect_audio_handler,
            frame_size_ms=frame_size_ms,
        )

        if isinstance(noise_cancellation, rtc.FrameProcessor):
            logger.warning(
                "a single noise cancellation processor is shared by every participant, which "
                "interleaves speakers through one stateful filter. pass a callable returning "
                "a new processor per participant instead",
                extra={"processor": type(noise_cancellation).__name__},
            )

    @override
    def add_participant(self, participant: rtc.RemoteParticipant | str) -> None:
        identity = (
            participant.identity if isinstance(participant, rtc.RemoteParticipant) else participant
        )
        if identity in self._streams:
            return

        stream = self._new_stream(
            on_stream_changed=functools.partial(self._child_stream_changed, identity)
        )
        self._streams[identity] = stream
        self._child_added(identity, stream)
        # last: subscribing to a published track calls back into _child_stream_changed
        stream.set_participant(participant)

    @override
    def remove_participant(self, identity: str) -> None:
        if (stream := self._streams.pop(identity, None)) is None:
            return

        self._child_removed(identity, stream)
        task = asyncio.create_task(stream.aclose())
        task.add_done_callback(self._tasks.discard)
        self._tasks.add(task)

    @override
    def set_participant(self, participant: rtc.RemoteParticipant | str | None) -> None:
        """Ignored: every accepted participant is listened to, linking only drives the outputs."""

    @override
    async def __anext__(self) -> rtc.AudioFrame:
        return await self._data_ch.__anext__()

    @override
    def on_attached(self) -> None:
        # not forwarded to the participant streams on purpose, see _send()
        self._attached = True

    @override
    def on_detached(self) -> None:
        self._attached = False

    @override
    async def aclose(self) -> None:
        for stream in self._streams.values():
            await stream.aclose()
        self._streams.clear()

        self._data_ch.close()
        await aio.cancel_and_wait(*self._tasks)
        self._tasks.clear()

    def _send(self, frame: rtc.AudioFrame) -> None:
        """Hand a frame to the session, dropping it while the input is detached.

        Dropping here rather than in the participant streams keeps them reading: a stream
        that stops delivering stalls the mixer and starves the active speaker detection.
        """
        if self._attached and not self._data_ch.closed:
            self._data_ch.send_nowait(frame)

    # the hooks below are how a subclass combines its children; each is a no-op here because
    # the group itself only owns their lifetime, not what the session ends up hearing

    def _child_added(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        """A participant's stream was created, before it subscribes to anything."""
        pass

    def _child_removed(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        """A participant's stream is about to be closed."""
        pass

    def _child_stream_changed(self, identity: str) -> None:
        """A participant's track was subscribed, replaced or lost."""
        pass


class _MixedParticipantAudioInput(_ParticipantAudioInputGroup):
    """Sums every participant's microphone, so overlapping speech is preserved."""

    def __init__(self, room: rtc.Room, **kwargs: Any) -> None:
        super().__init__(room, **kwargs)
        self._mixing: set[str] = set()
        self._mixer = rtc.AudioMixer(
            self._sample_rate,
            self._num_channels,
            blocksize=int(self._sample_rate * self._frame_size_ms / 1000),
            # must stay above the frame interval: the mixer pads a stream that misses this
            # deadline, so a live stream would get chopped by a tighter timeout
            stream_timeout_ms=max(100, self._frame_size_ms * 2),
        )
        self._forward_atask = asyncio.create_task(self._forward_mixed())

        # a muted track is not unpublished and its stream doesn't end, it just stops
        # delivering: without these the mixer would keep waiting on it every block
        self._room.on("track_muted", self._on_track_mute_changed)
        self._room.on("track_unmuted", self._on_track_mute_changed)

    @override
    async def aclose(self) -> None:
        self._room.off("track_muted", self._on_track_mute_changed)
        self._room.off("track_unmuted", self._on_track_mute_changed)

        await self._mixer.aclose()
        await aio.cancel_and_wait(self._forward_atask)
        self._mixing.clear()
        await super().aclose()

    @override
    def _child_removed(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        self._stop_mixing(identity, stream)

    @override
    def _child_stream_changed(self, identity: str) -> None:
        """Register a participant with the mixer only while it has audio to give.

        The mixer waits `stream_timeout_ms` on every registered stream for every block, so an
        idle one paces the whole mix below real time and warns on every block.
        """
        if (stream := self._streams.get(identity)) is None:
            return

        if not stream.streaming:
            self._stop_mixing(identity, stream)
        elif identity not in self._mixing:
            stream._drain()  # start from live audio, not from what piled up while unregistered
            self._mixer.add_stream(stream)
            self._mixing.add(identity)

    def _stop_mixing(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        if identity not in self._mixing:
            return

        self._mixer.remove_stream(stream)
        self._mixing.discard(identity)

        if not self._mixing:
            # nothing left to mix: the session needs silence to flush the pending transcript.
            # while others are still talking the mix carries on and injecting it would punch
            # a hole in their audio
            self._send(audio.silence_frame(_TURN_GAP, self._sample_rate, self._num_channels))

    def _on_track_mute_changed(
        self, participant: rtc.Participant, publication: rtc.TrackPublication
    ) -> None:
        self._child_stream_changed(participant.identity)

    @log_exceptions(logger=logger)
    async def _forward_mixed(self) -> None:
        async for frame in self._mixer:
            self._send(frame)


class _ActiveSpeakerAudioInput(_ParticipantAudioInputGroup):
    """Forwards one participant at a time, the one the server reports as speaking.

    Overlapping speech is dropped rather than summed, which keeps the audio the STT sees
    clean at the cost of losing whatever the other participants said meanwhile.
    """

    def __init__(self, room: rtc.Room, **kwargs: Any) -> None:
        super().__init__(room, **kwargs)
        self._speaking: str | None = None
        # `active_speakers_changed` lags the onset of speech, so a participant's audio is held
        # until they are selected and flushed as pre-roll: without it every turn loses its
        # first syllables. it doubles as the overlap window, so keep it short
        self._preroll: dict[str, deque[rtc.AudioFrame]] = {}
        self._preroll_len = max(1, int(_PREROLL / (self._frame_size_ms / 1000)))
        self._room.on("active_speakers_changed", self._on_active_speakers_changed)

    @override
    async def aclose(self) -> None:
        self._room.off("active_speakers_changed", self._on_active_speakers_changed)
        await super().aclose()
        self._preroll.clear()

    @override
    def _child_added(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        self._preroll[identity] = deque(maxlen=self._preroll_len)
        task = asyncio.create_task(self._forward_speaker(identity, stream))
        task.add_done_callback(self._tasks.discard)
        self._tasks.add(task)

    @override
    def _child_removed(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        self._preroll.pop(identity, None)
        if self._speaking == identity:
            self._release()

    def _on_active_speakers_changed(self, speakers: list[rtc.Participant]) -> None:
        # ponytail: selection is driven entirely by the server's speaker updates, so nothing is
        # forwarded if they never arrive. fall back to picking the loudest pre-roll by RMS if
        # that ever shows up in practice
        # the server sends the loudest first; identities we don't listen to are not candidates
        speaking = [p.identity for p in speakers if p.identity in self._streams]

        if self._speaking is not None and self._speaking not in speaking:
            self._release()

        if self._speaking is None and speaking:
            self._select(speaking[0])

    def _select(self, identity: str) -> None:
        self._speaking = identity
        for frame in self._preroll.get(identity, ()):
            self._send(frame)
        for preroll in self._preroll.values():
            preroll.clear()

    def _release(self) -> None:
        self._speaking = None
        # segment the turn: without a gap this speaker's tail and the next one's opening
        # reach the STT as a single utterance and end up merged in the transcript
        self._send(audio.silence_frame(_TURN_GAP, self._sample_rate, self._num_channels))

    @log_exceptions(logger=logger)
    async def _forward_speaker(self, identity: str, stream: _ParticipantAudioInputStream) -> None:
        async for frame in stream:
            if identity == self._speaking:
                self._send(frame)
            elif (preroll := self._preroll.get(identity)) is not None:
                preroll.append(frame)  # bounded: the oldest frame falls off


ROOM_AUDIO_INPUTS: dict[ParticipantsMode, Callable[..., _RoomAudioInput]] = {
    "linked": _ParticipantAudioInputStream,
    "mix": _MixedParticipantAudioInput,
    "pick": _ActiveSpeakerAudioInput,
}


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
