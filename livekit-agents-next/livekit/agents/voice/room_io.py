"""The room a session runs in: its options (``RoomOptions`` and the per-track input and output
options), and ``livekit.rtc`` as the core's room traits see it (``RtcRoom`` and
its participants, publications, sources and stream writers): each ``_core_<method>`` is one SDK
call. SDK enums cross as the core's names, read through ``_core.rtc_wire_names()``."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import asdict
from typing import TYPE_CHECKING, TypeAlias, TypedDict, cast

from livekit import rtc
from livekit.agents._generated.audio_errors import (
    AudioProcessingError as AudioProcessingError,
)
from livekit.agents._generated.room_api import (
    AudioInputOptions as AudioInputOptions,
    AudioOutputOptions as AudioOutputOptions,
    RoomOptions as RoomOptions,
    TextOutputOptions as TextOutputOptions,
    VideoInputOptions as VideoInputOptions,
)
from livekit.agents._generated.room_input_events import (
    NoiseCancellationParams as NoiseCancellationParams,
    TextInputEvent as TextInputEvent,
)

from .. import _core
from .._generated import rtc_wire
from .._generated.room_api import TextInputOptions as TextInputOptions
from ..types import ATTRIBUTE_PUBLISH_ON_BEHALF

if TYPE_CHECKING:
    from .agent_session import VoiceSession

__all__ = [
    "RoomOptions",
    "ATTRIBUTE_PUBLISH_ON_BEHALF",
    "TextInputEvent",
    "TextInputOptions",
    "AudioInputOptions",
    "AudioOutputOptions",
    "TextOutputOptions",
    "VideoInputOptions",
    "AudioProcessingError",
]

DEFAULT_CLOSE_ON_DISCONNECT_REASONS: list[rtc.DisconnectReason.ValueType] = [
    rtc.DisconnectReason.CLIENT_INITIATED,
    rtc.DisconnectReason.ROOM_DELETED,
    rtc.DisconnectReason.USER_REJECTED,
]


TextInputCallback = Callable[["VoiceSession", TextInputEvent], Coroutine[None, None, None] | None]


NoiseCancellationSelector: TypeAlias = Callable[
    [NoiseCancellationParams],
    rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None,
]


_CONNECTION, _DISCONNECT, _KINDS, _SOURCES, _FEATURES = _core.rtc_wire_names()


def _name(names: list[str], value: int, unknown: str) -> str:
    return names[value] if 0 <= value < len(names) else unknown


class Room:
    """An ``rtc.Room`` as the core sees it. Its SDK events are re-emitted into the core's event
    bus; one per room (``of``), so the handlers are registered once."""

    @classmethod
    def of(cls, room: rtc.Room) -> Room:
        adapter = getattr(room, "_lk_core_room", None)
        if adapter is None:
            adapter = cls(room)
            room._lk_core_room = adapter  # type: ignore[attr-defined]
        return adapter

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._bus = _core.RoomEventBus()
        self._rpc_router = _core._core_RoomRpcRouter()
        self._local = LocalParticipant(self)
        bus = self._bus
        room.on(
            "track_subscribed",
            lambda track, pub, p: _core._core_room_track_subscribed(
                bus, self._remote(p), self._publication(pub)
            ),
        )
        room.on(
            "track_unsubscribed",
            lambda track, pub, p: _core._core_room_track_unsubscribed(
                bus, self._remote(p), self._publication(pub)
            ),
        )
        room.on(
            "track_published",
            lambda pub, p: _core._core_room_track_published(
                bus, self._remote(p), self._publication(pub)
            ),
        )
        room.on(
            "track_unpublished",
            lambda pub, p: _core._core_room_track_unpublished(
                bus, self._remote(p), self._publication(pub)
            ),
        )
        room.on(
            "local_track_published",
            lambda pub, track: _core._core_room_local_track_published(bus, self._publication(pub)),
        )
        room.on(
            "participant_connected",
            lambda p: _core._core_room_participant_connected(bus, self._remote(p)),
        )
        room.on(
            "participant_active",
            lambda p: _core._core_room_participant_active(bus, self._remote(p)),
        )
        room.on("participant_attributes_changed", self._attributes_changed)
        room.on(
            "participant_disconnected",
            lambda p: _core._core_room_participant_disconnected(
                bus, self._remote(p), _name(_DISCONNECT, p.disconnect_reason, "unknown_reason")
            ),
        )
        room.on(
            "connection_state_changed",
            lambda state: _core._core_room_connection_state_changed(
                bus, _name(_CONNECTION, state, "disconnected")
            ),
        )
        room.on(
            "disconnected",
            lambda reason: _core._core_room_disconnected(
                bus, _name(_DISCONNECT, reason, "unknown_reason")
            ),
        )
        room.on(
            "sip_dtmf_received",
            lambda ev: _core._core_room_sip_dtmf_received(
                bus, ev.digit, ev.participant.identity if ev.participant is not None else None
            ),
        )

    def _attributes_changed(self, changed: list[str], participant: rtc.Participant) -> None:
        # the agent's own attributes are not a remote participant's
        if isinstance(participant, rtc.RemoteParticipant):
            _core._core_room_participant_attributes_changed(self._bus, self._remote(participant))

    def _remote(self, participant: rtc.RemoteParticipant) -> RemoteParticipant:
        return RemoteParticipant(self, participant)

    def _publication(self, publication: rtc.TrackPublication) -> TrackPublication:
        return TrackPublication(self, publication)

    def _core_rtc(self) -> Room:
        return self

    def _core_create_audio_source(self, opts: rtc_wire.TrackPublishOptions) -> AudioSource:
        return AudioSource(
            rtc.AudioSource(opts.sample_rate, opts.num_channels, queue_size_ms=opts.queue_size_ms)
        )

    def _core_create_video_source(self, opts: rtc_wire.VideoTrackOptions) -> VideoSource:
        return VideoSource(rtc.VideoSource(opts.width, opts.height))

    def _core_name(self) -> str:
        return self._room.name

    def _core_connection_state(self) -> str:
        return _name(_CONNECTION, self._room.connection_state, "disconnected")

    def _core_local(self) -> LocalParticipant:
        return self._local

    def _core_participant(self, identity: str) -> RemoteParticipant | None:
        participant = self._room.remote_participants.get(identity)
        return self._remote(participant) if participant is not None else None

    def _core_remote_participants(self) -> list[RemoteParticipant]:
        return [self._remote(p) for p in self._room.remote_participants.values()]

    def _core_event_bus(self) -> _core.RoomEventBus:
        return self._bus

    def _core_rpc_router(self) -> _core._core_RoomRpcRouter:
        return self._rpc_router

    def _core_register_byte_stream_handler(
        self, topic: str, streams: _core.ByteStreamSender
    ) -> None:
        def opened(reader: rtc.ByteStreamReader, sender: str) -> None:
            if not streams.open(reader.info, sender, reader):
                reader.close()

        self._room.register_byte_stream_handler(topic, opened)

    def _core_unregister_byte_stream_handler(self, topic: str) -> None:
        self._room.unregister_byte_stream_handler(topic)

    def _core_register_text_stream_handler(
        self, topic: str, streams: _core.TextStreamSender
    ) -> None:
        def opened(reader: rtc.TextStreamReader, sender: str) -> None:
            if not streams.open(reader.info, sender, reader):
                reader.close()

        self._room.register_text_stream_handler(topic, opened)

    def _core_unregister_text_stream_handler(self, topic: str) -> None:
        self._room.unregister_text_stream_handler(topic)


class _TextStreamSender(TypedDict, total=False):
    sender_identity: str


class LocalParticipant:
    def __init__(self, room: Room) -> None:
        self._room = room

    @property
    def _participant(self) -> rtc.LocalParticipant:
        return self._room._room.local_participant

    def _core_identity(self) -> str:
        return self._participant.identity

    def _core_kind(self) -> str:
        return _name(_KINDS, self._participant.kind, "agent")

    def _core_track_publications(self) -> list[TrackPublication]:
        return [self._room._publication(p) for p in self._participant.track_publications.values()]

    async def _core_publish_audio_source(
        self, source: AudioSource, opts: rtc_wire.TrackPublishOptions
    ) -> TrackPublication:
        track = rtc.LocalAudioTrack.create_audio_track(opts.name, source._source)
        publication = await self._participant.publish_track(
            track,
            rtc.TrackPublishOptions(
                source=cast(rtc.TrackSource.ValueType, _SOURCES.index(opts.source))
            ),
        )
        return self._room._publication(publication)

    async def _core_publish_video_source(
        self, source: VideoSource, opts: rtc_wire.VideoTrackOptions
    ) -> TrackPublication:
        track = rtc.LocalVideoTrack.create_video_track(opts.name, source._source)
        publication = await self._participant.publish_track(
            track,
            rtc.TrackPublishOptions(
                source=cast(rtc.TrackSource.ValueType, _SOURCES.index(opts.source))
            ),
        )
        return self._room._publication(publication)

    async def _core_publish_audio_track(
        self, opts: rtc_wire.TrackPublishOptions
    ) -> tuple[AudioSource, TrackPublication]:
        source = rtc.AudioSource(
            opts.sample_rate, opts.num_channels, queue_size_ms=opts.queue_size_ms
        )
        try:
            track = rtc.LocalAudioTrack.create_audio_track(opts.name, source)
            options = rtc.TrackPublishOptions(
                source=cast(rtc.TrackSource.ValueType, _SOURCES.index(opts.source))
            )
            publication = await self._participant.publish_track(track, options)
        except BaseException as error:
            try:
                await source.aclose()
            except BaseException as cleanup_error:
                error.add_note(f"Unpublished audio source cleanup failed: {cleanup_error!r}")
            raise
        return AudioSource(source), self._room._publication(publication)

    async def _core_unpublish_track(self, sid: str) -> None:
        await self._participant.unpublish_track(sid)

    async def _core_set_attributes(self, attributes: dict[str, str]) -> None:
        await self._participant.set_attributes(attributes)

    async def _core_publish_transcription(self, transcription: rtc_wire.Transcription) -> None:
        await self._participant.publish_transcription(
            rtc.Transcription(
                participant_identity=transcription.participant_identity,
                track_sid=transcription.track_sid,
                segments=[rtc.TranscriptionSegment(**asdict(s)) for s in transcription.segments],
            )
        )

    async def _core_stream_text(self, opts: rtc_wire.StreamTextOptions) -> TextStreamWriter:
        sender: _TextStreamSender = {}
        if opts.sender_identity is not None:
            sender["sender_identity"] = opts.sender_identity
        writer = await self._participant.stream_text(
            topic=opts.topic, attributes=opts.attributes, **sender
        )
        return TextStreamWriter(writer)

    async def _core_stream_bytes(self, opts: rtc_wire.StreamBytesOptions) -> ByteStreamWriter:
        writer = await self._participant.stream_bytes(
            name=opts.name,
            topic=opts.topic,
            destination_identities=opts.destination_identities,
            attributes=opts.attributes,
        )
        return ByteStreamWriter(writer)

    async def _core_perform_rpc(
        self, destination: str, method: str, payload: str
    ) -> tuple[int, str]:
        try:
            return 0, await self._participant.perform_rpc(
                destination_identity=destination, method=method, payload=payload
            )
        except rtc.RpcError as e:
            return e.code, e.message

    def _core_unregister_rpc_method(self, method: str) -> None:
        self._participant.unregister_rpc_method(method)

    async def _core_publish_dtmf(self, code: int, digit: str) -> None:
        await self._participant.publish_dtmf(code=code, digit=digit)

    def _core_register_rpc_method(self, method: str, handler: _core.RpcHandler) -> None:
        self._participant.register_rpc_method(
            method, lambda data: handler.call(data.caller_identity, data.payload)
        )


class ByteStreamWriter:
    def __init__(self, writer: rtc.ByteStreamWriter) -> None:
        self._writer = writer

    def _core_id(self) -> str:
        return self._writer.info.stream_id

    async def _core_write(self, data: bytes) -> None:
        await self._writer.write(data)

    async def _core_close(self) -> None:
        await self._writer.aclose()


class TextStreamWriter:
    def __init__(self, writer: rtc.TextStreamWriter) -> None:
        self._writer = writer

    async def _core_write(self, text: str) -> None:
        await self._writer.write(text)

    async def _core_close(self, attributes: dict[str, str]) -> None:
        await self._writer.aclose(attributes=attributes)


class AudioSource:
    """An ``rtc.AudioSource``, with the ledger of what it played and threw away."""

    def __init__(self, source: rtc.AudioSource) -> None:
        self._source = source
        self._ledger = _core.PlayoutLedger()

    async def _core_capture(self, frame: rtc.AudioFrame) -> None:
        await self._source.capture_frame(frame)
        self._ledger.add_pushed_samples(frame.samples_per_channel, frame.sample_rate)

    def _core_clear_queue(self) -> None:
        self._ledger.add_discarded(self._source.queued_duration)
        self._source.clear_queue()

    def _core_queued_duration(self) -> float:
        return self._source.queued_duration

    async def _core_wait_for_playout(self) -> rtc_wire.PlaybackInfo:
        await self._source.wait_for_playout()
        return self._ledger.settle()

    def _core_playback_info(self) -> rtc_wire.PlaybackInfo:
        return self._ledger.snapshot(self._source.queued_duration)

    async def _core_close(self) -> None:
        await self._source.aclose()


class VideoSource:
    """Direct SDK capture and close; pacing is owned by the native synchronizer."""

    def __init__(self, source: rtc.VideoSource) -> None:
        self._source = source

    def _core_capture(self, frame: rtc.VideoFrame) -> None:
        self._source.capture_frame(frame)

    async def _core_close(self) -> None:
        await self._source.aclose()


class RemoteParticipant:
    def __init__(self, room: Room, participant: rtc.RemoteParticipant) -> None:
        self._room = room
        self._participant = participant

    def _core_sid(self) -> str:
        return self._participant.sid

    def _core_identity(self) -> str:
        return self._participant.identity

    def _core_kind(self) -> str:
        return _name(_KINDS, self._participant.kind, "standard")

    def _core_active(self) -> bool:
        return self._participant.state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE

    def _core_attributes(self) -> dict[str, str]:
        return dict(self._participant.attributes)

    def _core_track_publications(self) -> list[TrackPublication]:
        return [self._room._publication(p) for p in self._participant.track_publications.values()]


class TrackPublication:
    """A local or remote ``rtc.TrackPublication``; each side answers its own half."""

    def __init__(self, room: Room, publication: rtc.TrackPublication) -> None:
        self._room = room
        self._publication = publication

    def _core_sid(self) -> str:
        return self._publication.sid

    def _core_name(self) -> str:
        return self._publication.name

    def _core_source(self) -> str:
        return _name(_SOURCES, self._publication.source, "unknown")

    def _core_kind(self) -> int:
        return self._publication.kind

    def _core_is_subscribed(self) -> bool:
        return self._publication.track is not None

    def _core_audio_features(self) -> list[str]:
        return [_FEATURES[f] for f in self._publication.audio_features if f < len(_FEATURES)]

    async def _core_set_subscribed(self, subscribed: bool) -> None:
        assert isinstance(self._publication, rtc.RemoteTrackPublication)
        self._publication.set_subscribed(subscribed)

    async def _core_wait_for_subscription(self) -> None:
        assert isinstance(self._publication, rtc.LocalTrackPublication)
        await self._publication.wait_for_subscription()

    def _core_open_audio_stream(self, opts: rtc_wire.AudioStreamOptions) -> _AudioStream:
        track = self._publication.track
        if track is None:
            raise RuntimeError(f"track {self._publication.sid} is no longer subscribed")
        noise_cancellation = opts.noise_cancellation
        owned = False
        if callable(noise_cancellation) and not isinstance(noise_cancellation, rtc.FrameProcessor):
            owned = True
            participant = self._room._room.remote_participants[opts.participant_identity]
            noise_cancellation = noise_cancellation(NoiseCancellationParams(participant, track))
        try:
            apm = (
                rtc.AudioProcessingModule(auto_gain_control=True)
                if opts.auto_gain_control
                else None
            )
            stream = rtc.AudioStream.from_track(
                track=track,
                capacity=opts.queue_capacity,
                sample_rate=opts.sample_rate,
                num_channels=opts.num_channels,
                frame_size_ms=opts.frame_size_ms,
                noise_cancellation=(
                    noise_cancellation
                    if isinstance(noise_cancellation, rtc.NoiseCancellationOptions)
                    else None
                ),
            )
        except BaseException as error:
            if owned and isinstance(noise_cancellation, rtc.FrameProcessor):
                try:
                    noise_cancellation._close()
                except BaseException as cleanup_error:
                    error.add_note(f"Audio input processor cleanup failed: {cleanup_error!r}")
            raise
        return _AudioStream(stream, noise_cancellation, apm, owns_processor=owned)

    def _core_open_video_stream(self) -> _VideoStream:
        track = self._publication.track
        if track is None:
            raise RuntimeError(f"track {self._publication.sid} is no longer subscribed")
        return _VideoStream(
            rtc.VideoStream.from_track(track=track, capacity=_core.video_queue_frames())
        )


class _AudioStream(AsyncIterator[rtc_wire.AudioStreamItem]):
    """One input track's frames, through its processors. A processor that fails on a frame is
    reported ahead of it, and the frame goes on as that processor received it."""

    def __init__(
        self,
        stream: rtc.AudioStream,
        noise_cancellation: rtc.NoiseCancellationOptions
        | rtc.FrameProcessor[rtc.AudioFrame]
        | None,
        apm: rtc.AudioProcessingModule | None,
        *,
        owns_processor: bool = False,
    ) -> None:
        self._stream = stream
        self._noise_cancellation = noise_cancellation
        self._apm = apm
        self._owned_processor = (
            noise_cancellation
            if owns_processor and isinstance(noise_cancellation, rtc.FrameProcessor)
            else None
        )
        self._ready: deque[rtc_wire.AudioStreamItem] = deque()

    def _core_frames(self) -> _AudioStream:
        return self

    async def _core_process_audio(
        self, frames: list[rtc.AudioFrame]
    ) -> list[rtc_wire.AudioStreamItem]:
        if isinstance(self._noise_cancellation, rtc.NoiseCancellationOptions):
            cause = RuntimeError("the SDK native audio filter does not support pre-connect replay")
            failed = _processor_failed(self._noise_cancellation, cause)
            return [
                item for f in frames for item in (failed, rtc_wire.AudioStreamItemFrame(frame=f))
            ]
        items: list[rtc_wire.AudioStreamItem] = []
        for frame in frames:
            items.extend(self._process_frame(frame))
        return items

    def _process_frame(self, frame: rtc.AudioFrame) -> list[rtc_wire.AudioStreamItem]:
        items: list[rtc_wire.AudioStreamItem] = []
        nc = self._noise_cancellation
        if isinstance(nc, rtc.FrameProcessor) and nc.enabled:
            # Processors may mutate their input before raising; keep the span they received.
            kept = _copy(frame)
            try:
                frame = nc._process(frame)
            except Exception as cause:
                items.append(_processor_failed(nc, cause))
                frame = kept
        if self._apm is not None:
            kept = _copy(frame)
            try:
                self._apm.process_stream(frame)
            except Exception as cause:
                items.append(_processor_failed(self._apm, cause))
                frame = kept
        items.append(rtc_wire.AudioStreamItemFrame(frame=frame))
        return items

    def __aiter__(self) -> _AudioStream:
        return self

    async def __anext__(self) -> rtc_wire.AudioStreamItem:
        if not self._ready:
            event = await self._stream.__anext__()
            self._ready.extend(self._process_frame(event.frame))
        return self._ready.popleft()

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            processor, self._owned_processor = self._owned_processor, None
            if processor is not None:
                processor._close()


def _copy(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        bytes(frame.data),
        frame.sample_rate,
        frame.num_channels,
        frame.samples_per_channel,
        userdata=frame.userdata,
    )


def _processor_failed(processor: object, cause: Exception) -> rtc_wire.AudioStreamItem:
    kind = type(processor)
    return rtc_wire.AudioStreamItemProcessorFailed(
        failure=rtc_wire.ProcessorFailure(
            label=f"{kind.__module__}.{kind.__qualname__}",
            processor=processor,
            error=cause,
            message=str(cause),
        )
    )


class _VideoStream(AsyncIterator[rtc_wire.VideoFrameRef]):
    def __init__(self, stream: rtc.VideoStream) -> None:
        self._stream = stream

    def __aiter__(self) -> _VideoStream:
        return self

    async def __anext__(self) -> rtc_wire.VideoFrameRef:
        event = await self._stream.__anext__()
        return rtc_wire.VideoFrameRef(
            frame=event.frame,
            width=event.frame.width,
            height=event.frame.height,
            timestamp_us=event.timestamp_us,
        )

    async def aclose(self) -> None:
        await self._stream.aclose()
