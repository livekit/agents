"""The room a session runs in: its options (``RoomOptions`` and the per-track input and output
options), and ``livekit.rtc`` as the core's room traits see it (``lk-agents-rtc``'s ``RtcRoom`` and
its participants, publications, sources and stream writers): each ``_core_<method>`` is one SDK
call. SDK enums cross as the core's names, read through ``_core.rtc_wire_names()``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, TypeVar, cast

from livekit import rtc

from .. import _core
from ..log import logger
from ..types import ATTRIBUTE_PUBLISH_ON_BEHALF, NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .io import TextOutput

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
]

DEFAULT_CLOSE_ON_DISCONNECT_REASONS: list[rtc.DisconnectReason.ValueType] = [
    rtc.DisconnectReason.CLIENT_INITIATED,
    rtc.DisconnectReason.ROOM_DELETED,
    rtc.DisconnectReason.USER_REJECTED,
]


@dataclass
class TextInputEvent:
    text: str
    info: rtc.TextStreamInfo | None = None
    participant: rtc.RemoteParticipant | None = None


TextInputCallback = Callable[["VoiceSession", TextInputEvent], Coroutine[None, None, None] | None]


@dataclass
class NoiseCancellationParams:
    participant: rtc.Participant
    track: rtc.Track


NoiseCancellationSelector: TypeAlias = Callable[
    [NoiseCancellationParams],
    rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None,
]


async def _default_text_input_cb(sess: VoiceSession, ev: TextInputEvent) -> None:
    async with sess._claim_user_turn():
        await sess.interrupt()
        sess.generate_reply(user_input=ev.text)


@dataclass
class TextInputOptions:
    text_input_cb: TextInputCallback = _default_text_input_cb


@dataclass
class AudioInputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    frame_size_ms: int = 50
    """The frame size in milliseconds for the audio input."""
    noise_cancellation: (
        rtc.NoiseCancellationOptions
        | NoiseCancellationSelector
        | rtc.FrameProcessor[rtc.AudioFrame]
        | None
    ) = None
    auto_gain_control: NotGivenOr[bool] = NOT_GIVEN
    """Enable automatic gain control (AGC) on the input audio.
    If not given, disabled when noise cancellation is configured directly.
    Set explicitly when using a noise cancellation selector."""
    pre_connect_audio: bool = True
    """Pre-connect audio enabled or not."""
    pre_connect_audio_timeout: float = 3.0
    """The pre-connect audio will be ignored if it doesn't arrive within this time."""


@dataclass
class VideoInputOptions:
    pass


@dataclass
class AudioOutputOptions:
    sample_rate: int = 24000
    num_channels: int = 1
    track_publish_options: rtc.TrackPublishOptions = field(
        default_factory=lambda: rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    track_name: NotGivenOr[str] = NOT_GIVEN
    """The name of the audio track to publish. If not provided, default to "roomio_audio"."""


@dataclass
class TextOutputOptions:
    sync_transcription: NotGivenOr[bool] = NOT_GIVEN
    """False to disable transcription synchronization with audio output.
    Otherwise, transcription is emitted as quickly as available."""
    transcription_speed_factor: float = 1.0
    """Speed factor of transcription synchronization with audio output.
    Only effective if `sync_transcription` is True."""
    next_in_chain: TextOutput | None = None
    """The next text output in the chain for the agent. If provided, the agent's transcription will be passed to it."""
    json_format: bool = False
    """Send the transcription as JSON dict for each chunk, including start and end timestamps if it's a TimedString."""


@dataclass
class RoomOptions:
    text_input: NotGivenOr[TextInputOptions | bool] = NOT_GIVEN
    """The text input options. If not provided, default to True."""
    audio_input: NotGivenOr[AudioInputOptions | bool] = NOT_GIVEN
    """The audio input options. If not provided, default to True."""
    video_input: NotGivenOr[VideoInputOptions | bool] = NOT_GIVEN
    """The video input options. If not provided, default to False."""
    audio_output: NotGivenOr[AudioOutputOptions | bool] = NOT_GIVEN
    """The audio output options. If not provided, default to True."""
    text_output: NotGivenOr[TextOutputOptions | bool] = NOT_GIVEN
    """The transcription output options. If not provided, default to True."""

    participant_kinds: NotGivenOr[list[rtc.ParticipantKind.ValueType]] = NOT_GIVEN
    """Participant kinds accepted for auto subscription. If not provided,
    accept `DEFAULT_PARTICIPANT_KINDS`."""
    participant_identity: NotGivenOr[str] = NOT_GIVEN
    """The participant to link to. If not provided, link to the first participant.
    Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`."""
    close_on_disconnect: bool = True
    """Close the VoiceSession if the linked participant disconnects with reasons in
    CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED."""
    delete_room_on_close: bool = False
    """Delete the room when the VoiceSession is closed, default to False"""

    def get_text_input_options(self) -> TextInputOptions | None:
        if isinstance(self.text_input, TextInputOptions):
            return self.text_input
        # if text_input is not given, default to enabled
        return TextInputOptions() if self.text_input is not False else None

    def get_audio_input_options(self) -> AudioInputOptions | None:
        if isinstance(self.audio_input, AudioInputOptions):
            return self.audio_input
        # if audio_input is not given, default to enabled
        return AudioInputOptions() if self.audio_input is not False else None

    def get_video_input_options(self) -> VideoInputOptions | None:
        if isinstance(self.video_input, VideoInputOptions):
            return self.video_input
        # if video_input is not given, default to disabled
        return VideoInputOptions() if self.video_input is True else None

    def get_audio_output_options(self) -> AudioOutputOptions | None:
        if isinstance(self.audio_output, AudioOutputOptions):
            return self.audio_output
        return AudioOutputOptions() if self.audio_output is not False else None

    def get_text_output_options(self) -> TextOutputOptions | None:
        if isinstance(self.text_output, TextOutputOptions):
            return self.text_output
        return TextOutputOptions() if self.text_output is not False else None

    def _to_core(self) -> dict[str, Any]:
        """The ``RoomIoConfig`` the core reads: every leg resolved, ``None`` when it is off."""
        audio_in = self.get_audio_input_options()
        audio_out = self.get_audio_output_options()
        text_out = self.get_text_output_options()
        return {
            "participant_identity": (
                self.participant_identity if is_given(self.participant_identity) else None
            ),
            "participant_kinds": (
                list(self.participant_kinds) if is_given(self.participant_kinds) else []
            ),
            "close_on_disconnect": self.close_on_disconnect,
            "delete_room_on_close": self.delete_room_on_close,
            "text_input": self.get_text_input_options() is not None,
            "video_input": self.get_video_input_options() is not None,
            "audio_input": audio_in
            and {
                "sample_rate": audio_in.sample_rate,
                "num_channels": audio_in.num_channels,
                "frame_size_ms": audio_in.frame_size_ms,
                "noise_cancellation": audio_in.noise_cancellation,
                # unsaid: off when noise cancellation is configured directly
                "auto_gain_control": (
                    audio_in.auto_gain_control
                    if is_given(audio_in.auto_gain_control)
                    else audio_in.noise_cancellation is None
                    or callable(audio_in.noise_cancellation)
                ),
                "pre_connect_audio": audio_in.pre_connect_audio,
                "pre_connect_audio_timeout": audio_in.pre_connect_audio_timeout,
            },
            "audio_output": audio_out
            and {
                "sample_rate": audio_out.sample_rate,
                "num_channels": audio_out.num_channels,
                "track_source": audio_out.track_publish_options.source,
                **({"track_name": audio_out.track_name} if is_given(audio_out.track_name) else {}),
            },
            "text_output": text_out
            and {
                "sync_transcription": (
                    text_out.sync_transcription if is_given(text_out.sync_transcription) else None
                ),
                "transcription_speed_factor": text_out.transcription_speed_factor,
                "json_format": text_out.json_format,
            },
        }


_CONNECTION, _DISCONNECT, _KINDS, _SOURCES, _FEATURES = _core.rtc_wire_names()
_Chunk = TypeVar("_Chunk")


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
        self._local = LocalParticipant(self)
        self._tasks: set[asyncio.Task[None]] = set()
        # SDK processing objects belonging to each live stream, also used for its buffered head.
        self._processors: dict[
            str,
            tuple[rtc.FrameProcessor[rtc.AudioFrame] | None, rtc.AudioProcessingModule | None],
        ] = {}
        bus = self._bus
        room.on(
            "track_subscribed",
            lambda track, pub, p: bus.track_subscribed(self._remote(p), self._publication(pub)),
        )
        room.on(
            "track_unsubscribed",
            lambda track, pub, p: bus.track_unsubscribed(self._remote(p), self._publication(pub)),
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
        room.on(
            "sip_dtmf_received",
            lambda ev: bus.sip_dtmf_received(
                ev.digit, ev.participant.identity if ev.participant is not None else None
            ),
        )

    def _attributes_changed(self, changed: list[str], participant: rtc.Participant) -> None:
        # the agent's own attributes are not a remote participant's
        if isinstance(participant, rtc.RemoteParticipant):
            self._bus.participant_attributes_changed(self._remote(participant))

    def _remote(self, participant: rtc.RemoteParticipant) -> RemoteParticipant:
        return RemoteParticipant(self, participant)

    def _publication(self, publication: rtc.TrackPublication) -> TrackPublication:
        return TrackPublication(self, publication)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

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

    def _forward_stream(
        self,
        reader: AsyncIterator[_Chunk],
        push: Callable[[_Chunk], bool],
        close_sink: Callable[[], None],
        close_reader: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        cleanup: asyncio.Task[None] | None = None
        cleanup_observed = False

        async def release() -> None:
            try:
                close_sink()
            finally:
                await close_reader()

        def close() -> asyncio.Task[None]:
            nonlocal cleanup
            if cleanup is None:
                cleanup = self._spawn(release())
            return cleanup

        async def pump() -> None:
            nonlocal cleanup_observed
            failure: BaseException | None = None
            try:
                async for chunk in reader:
                    if not push(chunk):
                        break
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    await asyncio.shield(close())
                except BaseException as cleanup_error:
                    cleanup_observed = not isinstance(cleanup_error, asyncio.CancelledError)
                    if failure is None:
                        raise
                    if cleanup_observed:
                        failure.add_note(f"Room stream cleanup failed: {cleanup_error!r}")

        def report_cleanup(task: asyncio.Task[None]) -> None:
            if not task.cancelled():
                error = task.exception()
                if error is not None and not cleanup_observed:
                    logger.error("Room stream cleanup failed", exc_info=error)

        task = self._spawn(pump())
        # Cancellation before the first poll never enters pump's finally block. A cleanup
        # failure after the reader ended has no caller left to observe it.
        task.add_done_callback(lambda _: close().add_done_callback(report_cleanup))

    def _core_register_byte_stream_handler(
        self, topic: str, streams: _core.ByteStreamSender
    ) -> None:
        def opened(reader: rtc.ByteStreamReader, sender: str) -> None:
            if (chunks := streams.open(reader.info, sender)) is None:
                reader.close()
                return
            self._forward_stream(reader, chunks.push, chunks.close, reader.aclose)

        self._room.register_byte_stream_handler(topic, opened)

    def _core_unregister_byte_stream_handler(self, topic: str) -> None:
        self._room.unregister_byte_stream_handler(topic)

    def _core_register_text_stream_handler(
        self, topic: str, streams: _core.TextStreamSender
    ) -> None:
        def opened(reader: rtc.TextStreamReader, sender: str) -> None:
            if (chunks := streams.open(reader.info, sender)) is None:
                reader.close()
                return
            self._forward_stream(reader, chunks.push, chunks.close, reader.aclose)

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

    async def _core_publish_audio_track(
        self, opts: _core.TrackPublishOptions
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

    async def _core_publish_transcription(self, transcription: _core.Transcription) -> None:
        await self._participant.publish_transcription(
            rtc.Transcription(
                participant_identity=transcription.participant_identity,
                track_sid=transcription.track_sid,
                segments=[rtc.TranscriptionSegment(**s) for s in transcription.segments],
            )
        )

    async def _core_stream_text(self, opts: _core.StreamTextOptions) -> TextStreamWriter:
        sender: _TextStreamSender = {}
        if opts.sender_identity is not None:
            sender["sender_identity"] = opts.sender_identity
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

    async def _core_publish_dtmf(self, code: int, digit: str) -> None:
        await self._participant.publish_dtmf(code=code, digit=digit)

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
        if (processing := self._room._processors.get(self._publication.sid)) is None:
            return frames
        processor, apm = processing
        processed = []
        for frame in frames:
            if processor is not None:
                frame = processor._process(frame)
            if apm is not None:
                apm.process_stream(frame)
            processed.append(frame)
        return processed

    def _core_open_audio_stream(self, opts: _core.AudioStreamOptions, frames: Any) -> None:
        track = self._publication.track
        if track is None:
            frames.close()
            raise RuntimeError(f"track {self._publication.sid} is no longer subscribed")
        sid = self._publication.sid
        noise_cancellation = opts.noise_cancellation
        # a selector picks per track; what it returns lives as long as this stream
        owned = False
        if callable(noise_cancellation) and not isinstance(noise_cancellation, rtc.FrameProcessor):
            owned = True
            participant = next(
                p
                for p in self._room._room.remote_participants.values()
                if sid in p.track_publications
            )
            noise_cancellation = noise_cancellation(NoiseCancellationParams(participant, track))
        processor = (
            noise_cancellation if isinstance(noise_cancellation, rtc.FrameProcessor) else None
        )
        stream = rtc.AudioStream.from_track(
            track=track,
            capacity=opts.queue_capacity,
            sample_rate=opts.sample_rate or 48000,
            num_channels=opts.num_channels or 1,
            frame_size_ms=opts.frame_size_ms,
            noise_cancellation=noise_cancellation,
            auto_close_noise_cancellation=owned,
        )
        apm = rtc.AudioProcessingModule(auto_gain_control=True) if opts.auto_gain_control else None
        processing = (processor, apm)
        self._room._processors[sid] = processing

        def push(event: rtc.AudioFrameEvent) -> bool:
            if apm is not None:
                apm.process_stream(event.frame)
            return bool(frames.push(event.frame))

        def close() -> None:
            try:
                frames.close()
            finally:
                if self._room._processors.get(sid) is processing:
                    self._room._processors.pop(sid, None)

        self._room._forward_stream(stream, push, close, stream.aclose)

    def _core_open_video_stream(self, frames: Any) -> None:
        track = self._publication.track
        if track is None:
            frames.close()
            raise RuntimeError(f"track {self._publication.sid} is no longer subscribed")
        stream = rtc.VideoStream.from_track(track=track, capacity=_core.video_queue_frames())

        def push(event: rtc.VideoFrameEvent) -> bool:
            frame = event.frame
            return bool(frames.push(frame, frame.width, frame.height, event.timestamp_us))

        self._room._forward_stream(stream, push, frames.close, stream.aclose)
