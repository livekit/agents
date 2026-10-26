"""``livekit.rtc`` as the core's room traits see it (``lk-agents-rtc``'s ``RtcRoom`` and its
participants, publications, sources and stream writers): each ``_core_<method>`` is one SDK call.
SDK enums cross as the core's names, read through ``_core.rtc_wire_names()``."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from livekit import rtc

from ... import _core

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
        return adapter  # type: ignore[no-any-return]

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._bus = _core.RoomEventBus()
        self._local = LocalParticipant(self)
        self._tasks: set[asyncio.Task[Any]] = set()
        bus = self._bus
        room.on(
            "track_subscribed",
            lambda track, pub, p: bus.track_subscribed(self._remote(p), self._publication(pub)),
        )
        room.on(
            "track_published",
            lambda pub, p: bus.track_published(self._remote(p), self._publication(pub)),
        )
        room.on(
            "track_unpublished",
            lambda pub, p: bus.track_unpublished(self._remote(p), self._publication(pub)),
        )
        room.on(
            "local_track_published",
            lambda pub, track: bus.local_track_published(self._publication(pub)),
        )
        room.on("participant_connected", lambda p: bus.participant_connected(self._remote(p)))
        room.on("participant_active", lambda p: bus.participant_active(self._remote(p)))
        room.on("participant_attributes_changed", self._attributes_changed)
        room.on(
            "participant_disconnected",
            lambda p: bus.participant_disconnected(
                self._remote(p), _name(_DISCONNECT, p.disconnect_reason, "unknown_reason")
            ),
        )
        room.on(
            "connection_state_changed",
            lambda state: bus.connection_state_changed(_name(_CONNECTION, state, "disconnected")),
        )
        room.on(
            "disconnected",
            lambda reason: bus.disconnected(_name(_DISCONNECT, reason, "unknown_reason")),
        )

    def _attributes_changed(self, changed: list[str], participant: rtc.Participant) -> None:
        # the agent's own attributes are not a remote participant's
        if isinstance(participant, rtc.RemoteParticipant):
            self._bus.participant_attributes_changed(self._remote(participant))

    def _remote(self, participant: rtc.RemoteParticipant) -> RemoteParticipant:
        return RemoteParticipant(self, participant)

    def _publication(self, publication: rtc.TrackPublication) -> TrackPublication:
        return TrackPublication(self, publication)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _core_rtc(self) -> Room:
        return self

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

    def _core_register_byte_stream_handler(self, topic: str, streams: Any) -> None:
        async def _pump(reader: rtc.ByteStreamReader, sender: str) -> None:
            if (chunks := streams.open(reader.info, sender)) is None:
                return
            async for chunk in reader:
                if not chunks.push(chunk):
                    break
            chunks.close()

        self._room.register_byte_stream_handler(
            topic, lambda reader, sender: self._spawn(_pump(reader, sender))
        )

    def _core_unregister_byte_stream_handler(self, topic: str) -> None:
        self._room.unregister_byte_stream_handler(topic)

    def _core_register_text_stream_handler(self, topic: str, streams: Any) -> None:
        async def _pump(reader: rtc.TextStreamReader, sender: str) -> None:
            if (chunks := streams.open(reader.info, sender)) is None:
                return
            async for chunk in reader:
                if not chunks.push(chunk):
                    break
            chunks.close()

        self._room.register_text_stream_handler(
            topic, lambda reader, sender: self._spawn(_pump(reader, sender))
        )

    def _core_unregister_text_stream_handler(self, topic: str) -> None:
        self._room.unregister_text_stream_handler(topic)


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

    async def _core_publish_audio_track(
        self, opts: _core.TrackPublishOptions
    ) -> tuple[AudioSource, TrackPublication]:
        source = rtc.AudioSource(
            opts.sample_rate, opts.num_channels, queue_size_ms=opts.queue_size_ms
        )
        track = rtc.LocalAudioTrack.create_audio_track(opts.name, source)
        options = rtc.TrackPublishOptions(source=_SOURCES.index(opts.source))
        publication = await self._participant.publish_track(track, options)
        return AudioSource(source), self._room._publication(publication)

    async def _core_unpublish_track(self, sid: str) -> None:
        await self._participant.unpublish_track(sid)

    async def _core_set_attributes(self, attributes: dict[str, str]) -> None:
        await self._participant.set_attributes(attributes)

    async def _core_publish_transcription(self, transcription: _core.Transcription) -> None:
        await self._participant.publish_transcription(
            rtc.Transcription(
                participant_identity=transcription.participant_identity,
                track_sid=transcription.track_sid,
                segments=[rtc.TranscriptionSegment(**s) for s in transcription.segments],
            )
        )

    async def _core_stream_text(self, opts: _core.StreamTextOptions) -> TextStreamWriter:
        sender = {} if opts.sender_identity is None else {"sender_identity": opts.sender_identity}
        writer = await self._participant.stream_text(
            topic=opts.topic, attributes=opts.attributes, **sender
        )
        return TextStreamWriter(writer)

    async def _core_stream_bytes(self, opts: _core.StreamBytesOptions) -> ByteStreamWriter:
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

    def _core_register_rpc_method(self, method: str, handler: Any) -> None:
        self._participant.register_rpc_method(
            method, lambda data: handler.call(data.caller_identity, data.payload)
        )


class ByteStreamWriter:
    def __init__(self, writer: rtc.ByteStreamWriter) -> None:
        self._writer = writer

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

    async def _core_wait_for_playout(self) -> _core.PlaybackInfo:
        await self._source.wait_for_playout()
        return self._ledger.settle()

    def _core_playback_info(self) -> _core.PlaybackInfo:
        return self._ledger.snapshot(self._source.queued_duration)

    async def _core_close(self) -> None:
        await self._source.aclose()


class RemoteParticipant:
    def __init__(self, room: Room, participant: rtc.RemoteParticipant) -> None:
        self._room = room
        self._participant = participant

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

    async def _core_process_audio(self, frames: list[rtc.AudioFrame]) -> list[rtc.AudioFrame]:
        return frames

    def _core_open_audio_stream(self, opts: _core.AudioStreamOptions, frames: Any) -> None:
        async def _pump() -> None:
            stream = rtc.AudioStream.from_track(
                track=self._publication.track,
                sample_rate=opts.sample_rate or 48000,
                num_channels=opts.num_channels or 1,
                frame_size_ms=opts.frame_size_ms,
                noise_cancellation=opts.noise_cancellation,
            )
            try:
                async for ev in stream:
                    if not frames.push(ev.frame):
                        break
            finally:
                frames.close()
                await stream.aclose()

        self._room._spawn(_pump())

    def _core_open_video_stream(self, frames: Any) -> None:
        async def _pump() -> None:
            stream = rtc.VideoStream.from_track(track=self._publication.track)
            try:
                async for ev in stream:
                    frame = ev.frame
                    if not frames.push(frame, frame.width, frame.height, ev.timestamp_us):
                        break
            finally:
                frames.close()
                await stream.aclose()

        self._room._spawn(_pump())
