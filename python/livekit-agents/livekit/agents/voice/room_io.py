from __future__ import annotations

import functools
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import TYPE_CHECKING, TypeAlias, cast

from livekit import rtc

from .. import _core
from .._generated import public_voice_room_io as _public, rtc_events, rtc_wire
from .._generated.public_voice_room_io import *  # noqa: F403
from ..types import ATTRIBUTE_PUBLISH_ON_BEHALF

if TYPE_CHECKING:
    from .agent_session import VoiceSession

__all__ = [
    "ATTRIBUTE_PUBLISH_ON_BEHALF",
]
__all__ += _public.__all__

TextInputCallback = Callable[["VoiceSession", TextInputEvent], Coroutine[None, None, None] | None]


NoiseCancellationSelector: TypeAlias = Callable[
    [NoiseCancellationParams],
    rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None,
]


class Room:
    # one adapter per rtc.Room, so its SDK event handlers are registered once
    sdk: rtc.Room
    bus: _core.RoomEventBus
    rpc_router: _core._core_RoomRpcRouter
    local: LocalParticipant

    def __new__(cls, room: rtc.Room) -> Room:
        adapter: Room | None = getattr(room, "_lk_core_room", None)
        if adapter is None:
            adapter = super().__new__(cls)
            adapter.sdk = room
            adapter.bus = _core.RoomEventBus()
            adapter.rpc_router = _core._core_RoomRpcRouter()
            adapter.local = LocalParticipant(adapter)
            rtc_events.subscribe(adapter)
            room._lk_core_room = adapter  # type: ignore[attr-defined]
        return adapter

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
        participant = self.sdk.remote_participants.get(identity)
        return self._remote(participant) if participant is not None else None

    def _core_remote_participants(self) -> list[RemoteParticipant]:
        return [self._remote(p) for p in self.sdk.remote_participants.values()]

    def _core_register_byte_stream_handler(
        self, topic: str, streams: _core.ByteStreamSender
    ) -> None:
        def opened(reader: rtc.ByteStreamReader, sender: str) -> None:
            if not streams.open(reader.info, sender, reader):
                reader.close()

        self.sdk.register_byte_stream_handler(topic, opened)

    def _core_register_text_stream_handler(
        self, topic: str, streams: _core.TextStreamSender
    ) -> None:
        def opened(reader: rtc.TextStreamReader, sender: str) -> None:
            if not streams.open(reader.info, sender, reader):
                reader.close()

        self.sdk.register_text_stream_handler(topic, opened)


class LocalParticipant:
    def __init__(self, room: Room) -> None:
        self._room = room

    @property
    def sdk(self) -> rtc.LocalParticipant:
        # resolved on each read: it exists only once the room connects
        return self._room.sdk.local_participant

    def _core_track_publications(self) -> list[TrackPublication]:
        return [self._room._publication(p) for p in self.sdk.track_publications.values()]

    async def _core_publish_audio_source(
        self, source: AudioSource, opts: rtc_wire.TrackPublishOptions
    ) -> TrackPublication:
        track = rtc.LocalAudioTrack.create_audio_track(opts.name, source.sdk)
        publication = await self.sdk.publish_track(
            track,
            rtc.TrackPublishOptions(source=cast(rtc.TrackSource.ValueType, opts.source)),
        )
        return self._room._publication(publication)

    async def _core_publish_video_source(
        self, source: rtc.VideoSource, opts: rtc_wire.VideoTrackOptions
    ) -> TrackPublication:
        track = rtc.LocalVideoTrack.create_video_track(opts.name, source)
        publication = await self.sdk.publish_track(
            track,
            rtc.TrackPublishOptions(source=cast(rtc.TrackSource.ValueType, opts.source)),
        )
        return self._room._publication(publication)

    async def _core_publish_audio_track(
        self, opts: rtc_wire.TrackPublishOptions
    ) -> rtc_wire.PublishedAudioTrack:
        source = rtc.AudioSource(
            opts.sample_rate, opts.num_channels, queue_size_ms=opts.queue_size_ms
        )
        try:
            track = rtc.LocalAudioTrack.create_audio_track(opts.name, source)
            options = rtc.TrackPublishOptions(source=cast(rtc.TrackSource.ValueType, opts.source))
            publication = await self.sdk.publish_track(track, options)
        except BaseException as error:
            try:
                await source.aclose()
            except BaseException as cleanup_error:
                error.add_note(f"Unpublished audio source cleanup failed: {cleanup_error!r}")
            raise
        return rtc_wire.PublishedAudioTrack(
            source=AudioSource(source), publication=self._room._publication(publication)
        )

    def _core_register_rpc_method(self, method: str, handler: _core.RpcHandler) -> None:
        self.sdk.register_rpc_method(
            method, lambda data: handler.call(data.caller_identity, data.payload)
        )


class AudioSource:
    def __init__(self, source: rtc.AudioSource) -> None:
        self.sdk = source
        self._ledger = _core.PlayoutLedger()

    async def _core_capture(self, frame: rtc.AudioFrame) -> None:
        await self.sdk.capture_frame(frame)
        self._ledger.add_pushed_samples(frame.samples_per_channel, frame.sample_rate)

    def _core_clear_queue(self) -> None:
        self._ledger.add_discarded(self.sdk.queued_duration)
        self.sdk.clear_queue()

    async def _core_wait_for_playout(self) -> rtc_wire.PlaybackInfo:
        await self.sdk.wait_for_playout()
        return self._ledger.settle()

    def _core_playback_info(self) -> rtc_wire.PlaybackInfo:
        return self._ledger.snapshot(self.sdk.queued_duration)


class RemoteParticipant:
    def __init__(self, room: Room, participant: rtc.RemoteParticipant) -> None:
        self._room = room
        self.sdk = participant

    def _core_track_publications(self) -> list[TrackPublication]:
        return [self._room._publication(p) for p in self.sdk.track_publications.values()]


class TrackPublication:
    def __init__(self, room: Room, publication: rtc.TrackPublication) -> None:
        self._room = room
        self.sdk = publication

    def _core_open_audio_stream(self, opts: rtc_wire.AudioStreamOptions) -> _AudioStream:
        track = self.sdk.track
        if track is None:
            raise RuntimeError(f"track {self.sdk.sid} is no longer subscribed")
        noise_cancellation: rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None
        selector = opts.noise_cancellation
        owned = False
        if (
            isinstance(selector, rtc.NoiseCancellationOptions | rtc.FrameProcessor)
            or selector is None
        ):
            noise_cancellation = selector
        elif callable(selector):
            owned = True
            participant = self._room.sdk.remote_participants[opts.participant_identity]
            chosen = selector(NoiseCancellationParams(participant, track))
            if not (
                isinstance(chosen, rtc.NoiseCancellationOptions | rtc.FrameProcessor)
                or chosen is None
            ):
                raise TypeError(f"noise_cancellation returned {type(chosen).__name__}")
            noise_cancellation = chosen
        else:
            raise TypeError(f"noise_cancellation is {type(selector).__name__}")
        try:
            apm = (
                rtc.AudioProcessingModule(auto_gain_control=True)
                if opts.auto_gain_control
                else None
            )
            native: rtc.NoiseCancellationOptions | rtc.FrameProcessor[rtc.AudioFrame] | None
            if isinstance(noise_cancellation, rtc.FrameProcessor):
                native = _RoomContext(noise_cancellation)
            elif isinstance(noise_cancellation, rtc.NoiseCancellationOptions):
                native = noise_cancellation
            else:
                native = None
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
        stream = _AudioStream(open_stream, noise_cancellation, apm, owns_processor=owned)
        # Open on the room's loop now, not at the first read: while the reader waits for the
        # pre-connect buffer the live audio queues, so speech in that wait is not lost.
        self._room.sdk._loop.call_soon_threadsafe(stream._open_now)
        return stream

    def _core_open_video_stream(self) -> _VideoStream:
        track = self.sdk.track
        if track is None:
            raise RuntimeError(f"track {self.sdk.sid} is no longer subscribed")
        return _VideoStream(
            functools.partial(
                rtc.VideoStream.from_track, track=track, capacity=_core.video_queue_frames()
            )
        )


class _AudioStream(AsyncIterator[rtc_wire.AudioStreamItem]):
    # The SDK stream opens on the first read: the core creates this off the event loop, and an
    # SDK stream binds the loop it is created on. A failing processor is reported ahead of the
    # frame, which continues as that processor received it.
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
        self._closed = False

    def _open_now(self) -> None:
        if self._stream is None and not self._closed:
            self._stream = self._open()

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
            # processors may mutate their input before raising; keep a copy of what they received
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
            self._open_now()
            assert self._stream is not None
            event = await self._stream.__anext__()
            self._ready.extend(self._process_frame(event.frame))
        return self._ready.popleft()

    async def aclose(self) -> None:
        self._closed = True
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


class _RoomContext(rtc.FrameProcessor[rtc.AudioFrame]):
    """Hands a processor the room context the SDK gives the processors it runs: the stream's
    room, participant and publication, and the room's credentials as they refresh (cloud models
    such as Krisp and BVC authenticate with them). Disabled, so the SDK never processes frames:
    the session runs the processor itself to report its failures and process pre-connect audio.
    """

    def __init__(self, processor: rtc.FrameProcessor[rtc.AudioFrame]) -> None:
        self._processor = processor

    @property
    def enabled(self) -> bool:
        return False

    @enabled.setter
    def enabled(self, value: bool) -> None:
        pass

    def _on_stream_info_updated(
        self, *, room_name: str, participant_identity: str, publication_sid: str
    ) -> None:
        self._processor._on_stream_info_updated(
            room_name=room_name,
            participant_identity=participant_identity,
            publication_sid=publication_sid,
        )

    def _on_stream_info_cleared(self) -> None:
        self._processor._on_stream_info_cleared()

    def _on_credentials_updated(self, *, token: str, url: str) -> None:
        self._processor._on_credentials_updated(token=token, url=url)

    def _on_credentials_cleared(self) -> None:
        self._processor._on_credentials_cleared()

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        return frame

    def _close(self) -> None:
        pass


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


class _VideoStream(AsyncIterator[rtc.VideoFrameEvent]):
    # opens the SDK stream on the first read, like `_AudioStream`
    def __init__(self, open_stream: Callable[[], rtc.VideoStream]) -> None:
        self._open = open_stream
        self._stream: rtc.VideoStream | None = None

    def __aiter__(self) -> _VideoStream:
        return self

    async def __anext__(self) -> rtc.VideoFrameEvent:
        if self._stream is None:
            self._stream = self._open()
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        if self._stream is not None:
            await self._stream.aclose()
