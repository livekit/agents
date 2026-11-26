import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from livekit import rtc

from ... import utils
from ...log import logger

PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"

# the stream and its attributes come from any room participant, so what one stream and one
# sender can make the job hold is capped; a client buffers only the seconds before the agent joins
_MAX_BUFFER_DURATION_S = 30.0
_MAX_STREAM_BYTES = 10 * 1024 * 1024
# a participant publishes one microphone track; streams for other track ids are never consumed
_MAX_BUFFERS_PER_PARTICIPANT = 2


@dataclass
class _PreConnectAudioBuffer:
    timestamp: float
    frames: list[rtc.AudioFrame] = field(default_factory=list)


class PreConnectAudioHandler:
    def __init__(self, room: rtc.Room, *, timeout: float, max_delta_s: float = 1.0):
        self._room = room
        self._timeout = timeout
        self._max_delta_s = max_delta_s

        # track id -> buffer
        self._buffers: dict[str, asyncio.Future[_PreConnectAudioBuffer]] = {}
        # track id -> identity of the participant that sent its stream, oldest first
        self._senders: dict[str, str] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

        self._registered_after_connect = False

    def register(self) -> None:
        def _handler(reader: rtc.ByteStreamReader, participant_id: str) -> None:
            task = asyncio.create_task(self._read_audio_task(reader, participant_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            def _on_timeout() -> None:
                logger.warning(
                    "pre-connect audio received but not completed in time",
                    extra={"participant": participant_id},
                )
                if not task.done():
                    task.cancel()

            timeout_handle = asyncio.get_event_loop().call_later(self._timeout, _on_timeout)
            task.add_done_callback(lambda _: timeout_handle.cancel())

        try:
            if self._room.isconnected():
                self._registered_after_connect = True
            self._room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, _handler)
        except ValueError:
            logger.warning(
                f"pre-connect audio handler for {PRE_CONNECT_AUDIO_BUFFER_STREAM} "
                "already registered, ignoring"
            )

    async def aclose(self) -> None:
        self._room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM)
        await utils.aio.cancel_and_wait(*self._tasks)
        self._buffers.clear()
        self._senders.clear()

    async def wait_for_data(self, track_id: str) -> list[rtc.AudioFrame]:
        # the handler is enabled by default, log a warning only if the buffer is actually used
        if self._registered_after_connect:
            logger.warning(
                "pre-connect audio handler registered after room connection, "
                "start RoomIO before ctx.connect() to ensure seamless audio buffer.",
                extra={"track_id": track_id},
            )

        self._buffers.setdefault(track_id, asyncio.Future())
        fut = self._buffers[track_id]

        try:
            if fut.done():
                buf = fut.result()
                if (delta := time.time() - buf.timestamp) > self._max_delta_s:
                    logger.warning(
                        "pre-connect audio buffer is too old",
                        extra={"track_id": track_id, "delta_time": delta},
                    )
                    return []
                return buf.frames

            buf = await asyncio.wait_for(fut, self._timeout)
            return buf.frames
        finally:
            # aclose() or a capped sender may already have dropped it
            self._buffers.pop(track_id, None)
            self._senders.pop(track_id, None)

    def _make_room_for(self, participant_id: str, track_id: str) -> bool:
        owned = [t for t, p in self._senders.items() if p == participant_id and t != track_id]
        if len(owned) < _MAX_BUFFERS_PER_PARTICIPANT:
            return True

        for t in owned:
            if (fut := self._buffers.get(t)) is None or fut.done():
                self._buffers.pop(t, None)
                self._senders.pop(t, None)
                return True

        return False

    @utils.log_exceptions(logger=logger)
    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str) -> None:
        try:
            await self._read_audio(reader, participant_id)
        finally:
            # a reader left unread keeps queueing whatever the sender streams until it ends
            reader.close()

    async def _read_audio(self, reader: rtc.ByteStreamReader, participant_id: str) -> None:
        if not reader.info.attributes or not (track_id := reader.info.attributes.get("trackId")):
            logger.warning(
                "pre-connect audio received but no trackId", extra={"participant": participant_id}
            )
            return

        if not self._make_room_for(participant_id, track_id):
            logger.warning(
                "too many pre-connect audio streams in flight, ignoring",
                extra={"participant": participant_id, "track_id": track_id},
            )
            return

        if (fut := self._buffers.get(track_id)) and fut.done():
            # reset the buffer if it's already set
            self._buffers.pop(track_id)
        self._senders.pop(track_id, None)
        self._senders[track_id] = participant_id
        self._buffers.setdefault(track_id, asyncio.Future())
        fut = self._buffers[track_id]

        buf = _PreConnectAudioBuffer(timestamp=time.time())
        try:
            if (
                "sampleRate" not in reader.info.attributes
                or "channels" not in reader.info.attributes
            ):
                raise ValueError("sampleRate or channels not found in pre-connect byte stream")

            sample_rate = int(reader.info.attributes["sampleRate"])
            num_channels = int(reader.info.attributes["channels"])

            duration: float = 0
            received = 0
            decoded = 0

            def _check_limits(chunk_len: int = 0) -> None:
                nonlocal received
                received += chunk_len
                if received > _MAX_STREAM_BYTES or decoded > _MAX_STREAM_BYTES:
                    raise ValueError("pre-connect audio stream exceeds the size limit")
                if duration > _MAX_BUFFER_DURATION_S:
                    raise ValueError("pre-connect audio stream exceeds the duration limit")

            # check if we need to decode opus
            is_opus = False
            if reader.info.mime_type:
                # JS may send "mime_type" as "audio/opus" or "audio/webm;codecs=opus"
                is_opus = (
                    reader.info.mime_type == "audio/opus" or "codecs=opus" in reader.info.mime_type
                )

            if is_opus:
                decoder = utils.codecs.AudioStreamDecoder(
                    sample_rate=sample_rate, num_channels=num_channels
                )

                async def _collect_decoded() -> None:
                    nonlocal duration, decoded
                    async for decoded_frame in decoder:
                        buf.frames.append(decoded_frame)
                        duration += decoded_frame.duration
                        decoded += len(decoded_frame.data) * 2
                        _check_limits()

                # the decoder thread runs ahead of its input without backpressure, and a few
                # KiB of opus can expand to hours of PCM at a sender-chosen sample rate, so
                # decoded output is checked while input is still being pushed
                collect = asyncio.create_task(_collect_decoded())
                try:
                    async for chunk in reader:
                        if collect.done():
                            break
                        _check_limits(len(chunk))
                        decoder.push(chunk)

                    decoder.end_input()
                    await collect
                finally:
                    # the decoder runs on its own thread, which stays blocked until closed
                    await decoder.aclose()
                    await utils.aio.cancel_and_wait(collect)
            else:
                # Process raw audio directly through AudioByteStream
                audio_stream = utils.audio.AudioByteStream(sample_rate, num_channels)
                async for chunk in reader:
                    _check_limits(len(chunk))
                    for frame in audio_stream.push(chunk):
                        buf.frames.append(frame)
                        duration += frame.duration
                    _check_limits()

                # Get any remaining frames
                for frame in audio_stream.flush():
                    buf.frames.append(frame)
                    duration += frame.duration

            logger.debug(
                "pre-connect audio received",
                extra={
                    "duration": duration,
                    "track_id": track_id,
                    "participant": participant_id,
                    "channels": num_channels,
                    "sample_rate": sample_rate,
                },
            )

            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(buf)
        except asyncio.CancelledError:
            # a pending future would keep counting against the sender's cap forever
            _fail_buffer(
                fut, buf, asyncio.TimeoutError("pre-connect audio stream not completed in time")
            )
            raise
        except Exception as e:
            _fail_buffer(fut, buf, e)


def _fail_buffer(
    fut: asyncio.Future[_PreConnectAudioBuffer], buf: _PreConnectAudioBuffer, exc: BaseException
) -> None:
    # the stored exception's traceback keeps buf alive until the future is dropped
    buf.frames.clear()
    if not fut.done():
        fut.set_exception(exc)
        # nothing may ever wait on a stream sent under a made-up track id, and asyncio logs
        # an error for every such future that is garbage-collected unretrieved
        fut.exception()
