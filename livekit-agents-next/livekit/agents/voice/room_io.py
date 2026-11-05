"""The room a session runs in: its options (``RoomOptions`` and the per-track input and output
options), and ``livekit.rtc`` as the core's room traits see it (``RtcRoom`` and its participants,
publications and sources). Each adapter holds its SDK object as ``rtc``: the core's generated
proxies read and call the SDK through it, and each ``_core_<method>`` here is an SDK call that
builds or converts more than a name. Stream writers and video sources are SDK objects the core
calls directly. SDK enums reach the core's events as the names ``_core.rtc_wire_names()`` lists."""

from __future__ import annotations

import functools
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


_CONNECTION, _DISCONNECT, _SOURCES = _core.rtc_wire_names()


def _name(names: list[str], value: int, unknown: str) -> str:
    return names[value] if 0 <= value < len(names) else unknown


class Room:
    """An ``rtc.Room`` (``rtc``) as the core's ``RtcRoom``: its event bus (``bus``), RPC router
    and local participant. Its SDK events are re-emitted into the bus; one per room (``of``), so
    the handlers are registered once. The core reads the SDK room's attributes through ``rtc``;
    the methods here are the calls that build SDK objects or register SDK handlers."""

    @classmethod
    def of(cls, room: rtc.Room) -> Room:
        adapter = getattr(room, "_lk_core_room", None)
        if adapter is None:
            adapter = cls(room)
            room._lk_core_room = adapter  # type: ignore[attr-defined]
        return adapter

    def __init__(self, room: rtc.Room) -> None:
        self.rtc = room
        self.bus = _core.RoomEventBus()
        self.rpc_router = _core._core_RoomRpcRouter()
        self.local = LocalParticipant(self)
        bus = self.bus
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
            _core._core_room_participant_attributes_changed(self.bus, self._remote(participant))

    def _remote(self, participant: rtc.RemoteParticipant) -> RemoteParticipant:
        return RemoteParticipant(self, participant)

    def _publication(self, publication: rtc.TrackPublication) -> TrackPublication:
        return TrackPublication(self, publication)

    def _core_create_audio_source(self, opts: rtc_wire.TrackPublishOptions) -> AudioSource:
        return AudioSource(
            rtc.AudioSource(opts.sample_rate, opts.num_channels, queue_size_ms=opts.queue_size_ms)
        )

    def _core_create_video_source(self, opts: rtc_wire.VideoTrackOptions) -> rtc.VideoSource:
        return rtc.VideoSource(opts.width, opts.height)

    def _core_participant(self, identity: str) -> RemoteParticipant | None:
        participant = self.rtc.remote_participants.get(identity)
        return self._remote(participant) if participant is not None else None

    def _core_remote_participants(self) -> list[RemoteParticipant]:
        return [self._remote(p) for p in self.rtc.remote_participants.values()]

    def _core_register_byte_stream_handler(
        self, topic: str, streams: _core.ByteStreamSender
    ) -> None:
        def opened(reader: rtc.ByteStreamReader, sender: str) -> None:
            if not streams.open(reader.info, sender, reader):
                reader.close()

        self.rtc.register_byte_stream_handler(topic, opened)

    def _core_register_text_stream_handler(
        self, topic: str, streams: _core.TextStreamSender
    ) -> None:
        def opened(reader: rtc.TextStreamReader, sender: str) -> None:
            if not streams.open(reader.info, sender, reader):
                reader.close()

        self.rtc.register_text_stream_handler(topic, opened)


class _TextStreamSender(TypedDict, total=False):
    sender_identity: str


class LocalParticipant:
    """The room's ``rtc.LocalParticipant`` (``rtc``, resolved on each read: it exists once the
    room connects) as the core's ``LocalParticipant``. The methods here build SDK tracks,
    transcriptions, streams and RPC handlers."""

    def __init__(self, room: Room) -> None:
        self._room = room

    @property
    def rtc(self) -> rtc.LocalParticipant:
        return self._room.rtc.local_participant

    def _core_track_publications(self) -> list[TrackPublication]:
        return [self._room._publication(p) for p in self.rtc.track_publications.values()]

    async def _core_publish_audio_source(
        self, source: AudioSource, opts: rtc_wire.TrackPublishOptions
    ) -> TrackPublication:
        track = rtc.LocalAudioTrack.create_audio_track(opts.name, source.rtc)
        publication = await self.rtc.publish_track(
            track,
            rtc.TrackPublishOptions(
                source=cast(rtc.TrackSource.ValueType, _SOURCES.index(opts.source))
            ),
        )
        return self._room._publication(publication)

    async def _core_publish_video_source(
        self, source: rtc.VideoSource, opts: rtc_wire.VideoTrackOptions
    ) -> TrackPublication:
        track = rtc.LocalVideoTrack.create_video_track(opts.name, source)
        publication = await self.rtc.publish_track(
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
            publication = await self.rtc.publish_track(track, options)
        except BaseException as error:
            try:
                await source.aclose()
            except BaseException as cleanup_error:
                error.add_note(f"Unpublished audio source cleanup failed: {cleanup_error!r}")
            raise
        return AudioSource(source), self._room._publication(publication)

    async def _core_publish_transcription(self, transcription: rtc_wire.Transcription) -> None:
        await self.rtc.publish_transcription(
            rtc.Transcription(
                participant_identity=transcription.participant_identity,
                track_sid=transcription.track_sid,
                segments=[rtc.TranscriptionSegment(**asdict(s)) for s in transcription.segments],
            )
        )

    async def _core_stream_text(self, opts: rtc_wire.StreamTextOptions) -> rtc.TextStreamWriter:
        sender: _TextStreamSender = {}
        if opts.sender_identity is not None:
            sender["sender_identity"] = opts.sender_identity
        return await self.rtc.stream_text(topic=opts.topic, attributes=opts.attributes, **sender)

    async def _core_stream_bytes(self, opts: rtc_wire.StreamBytesOptions) -> rtc.ByteStreamWriter:
        return await self.rtc.stream_bytes(
            name=opts.name,
            topic=opts.topic,
            destination_identities=opts.destination_identities,
            attributes=opts.attributes,
        )

    async def _core_perform_rpc(
        self, destination: str, method: str, payload: str
    ) -> tuple[int, str]:
        try:
            return 0, await self.rtc.perform_rpc(
                destination_identity=destination, method=method, payload=payload
            )
        except rtc.RpcError as e:
            return e.code, e.message

    def _core_register_rpc_method(self, method: str, handler: _core.RpcHandler) -> None:
        self.rtc.register_rpc_method(
            method, lambda data: handler.call(data.caller_identity, data.payload)
        )


class AudioSource:
    """An ``rtc.AudioSource`` (``rtc``), with the ledger of what it played and threw away."""

    def __init__(self, source: rtc.AudioSource) -> None:
        self.rtc = source
        self._ledger = _core.PlayoutLedger()

    async def _core_capture(self, frame: rtc.AudioFrame) -> None:
        await self.rtc.capture_frame(frame)
        self._ledger.add_pushed_samples(frame.samples_per_channel, frame.sample_rate)

    def _core_clear_queue(self) -> None:
        self._ledger.add_discarded(self.rtc.queued_duration)
        self.rtc.clear_queue()

    async def _core_wait_for_playout(self) -> rtc_wire.PlaybackInfo:
        await self.rtc.wait_for_playout()
        return self._ledger.settle()

    def _core_playback_info(self) -> rtc_wire.PlaybackInfo:
        return self._ledger.snapshot(self.rtc.queued_duration)


class RemoteParticipant:
    """An ``rtc.RemoteParticipant`` (``rtc``) of ``room``, whose publications it hands out."""

    def __init__(self, room: Room, participant: rtc.RemoteParticipant) -> None:
        self._room = room
        self.rtc = participant

    def _core_active(self) -> bool:
        return self.rtc.state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE

    def _core_track_publications(self) -> list[TrackPublication]:
        return [self._room._publication(p) for p in self.rtc.track_publications.values()]


class TrackPublication:
    """A local or remote ``rtc.TrackPublication`` (``rtc``) of ``room``; each side answers its own
    half. The methods here open its SDK streams and change its subscription."""

    def __init__(self, room: Room, publication: rtc.TrackPublication) -> None:
        self._room = room
        self.rtc = publication

    def _core_is_subscribed(self) -> bool:
        return self.rtc.track is not None

    async def _core_set_subscribed(self, subscribed: bool) -> None:
        assert isinstance(self.rtc, rtc.RemoteTrackPublication)
        self.rtc.set_subscribed(subscribed)

    def _core_open_audio_stream(self, opts: rtc_wire.AudioStreamOptions) -> _AudioStream:
        track = self.rtc.track
        if track is None:
            raise RuntimeError(f"track {self.rtc.sid} is no longer subscribed")
        noise_cancellation = opts.noise_cancellation
        owned = False
        if callable(noise_cancellation) and not isinstance(noise_cancellation, rtc.FrameProcessor):
            owned = True
            participant = self._room.rtc.remote_participants[opts.participant_identity]
            noise_cancellation = noise_cancellation(NoiseCancellationParams(participant, track))
        try:
            apm = (
                rtc.AudioProcessingModule(auto_gain_control=True)
                if opts.auto_gain_control
                else None
            )
            native = (
                noise_cancellation
                if isinstance(noise_cancellation, rtc.NoiseCancellationOptions)
                else None
            )
            open_stream = functools.partial(
                rtc.AudioStream.from_track,
                track=track,
                capacity=max(1, opts.queue_size_ms // (opts.frame_size_ms or 10)),
                sample_rate=opts.sample_rate,
                num_channels=opts.num_channels,
                frame_size_ms=opts.frame_size_ms,
                noise_cancellation=native,
            )
        except BaseException as error:
            if owned and isinstance(noise_cancellation, rtc.FrameProcessor):
                try:
                    noise_cancellation._close()
                except BaseException as cleanup_error:
                    error.add_note(f"Audio input processor cleanup failed: {cleanup_error!r}")
            raise
        return _AudioStream(open_stream, noise_cancellation, apm, owns_processor=owned)

    def _core_open_video_stream(self) -> _VideoStream:
        track = self.rtc.track
        if track is None:
            raise RuntimeError(f"track {self.rtc.sid} is no longer subscribed")
        return _VideoStream(
            functools.partial(
                rtc.VideoStream.from_track, track=track, capacity=_core.video_queue_frames()
            )
        )


class _AudioStream(AsyncIterator[rtc_wire.AudioStreamItem]):
    """One input track's frames, through its processors. A processor that fails on a frame is
    reported ahead of it, and the frame goes on as that processor received it.

    The SDK stream opens on the first read, on the event loop: the core opens this from its own
    thread, and an SDK stream binds the loop it is created on."""

    def __init__(
        self,
        open_stream: Callable[[], rtc.AudioStream],
        noise_cancellation: rtc.NoiseCancellationOptions
        | rtc.FrameProcessor[rtc.AudioFrame]
        | None,
        apm: rtc.AudioProcessingModule | None,
        *,
        owns_processor: bool = False,
    ) -> None:
        self._open = open_stream
        self._stream: rtc.AudioStream | None = None
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
            if self._stream is None:
                self._stream = self._open()
            event = await self._stream.__anext__()
            self._ready.extend(self._process_frame(event.frame))
        return self._ready.popleft()

    async def aclose(self) -> None:
        try:
            if self._stream is not None:
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
    """One input track's frames; the SDK stream opens on the first read, as `_AudioStream`."""

    def __init__(self, open_stream: Callable[[], rtc.VideoStream]) -> None:
        self._open = open_stream
        self._stream: rtc.VideoStream | None = None

    def __aiter__(self) -> _VideoStream:
        return self

    async def __anext__(self) -> rtc_wire.VideoFrameRef:
        if self._stream is None:
            self._stream = self._open()
        event = await self._stream.__anext__()
        return rtc_wire.VideoFrameRef(
            frame=event.frame,
            width=event.frame.width,
            height=event.frame.height,
            timestamp_us=event.timestamp_us,
        )

    async def aclose(self) -> None:
        if self._stream is not None:
            await self._stream.aclose()
