import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from livekit import rtc

from ... import utils
from ...log import logger

PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"


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
            self._buffers.pop(track_id)

    @utils.log_exceptions(logger=logger)
    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str) -> None:
        if not reader.info.attributes or not (track_id := reader.info.attributes.get("trackId")):
            logger.warning(
                "pre-connect audio received but no trackId", extra={"participant": participant_id}
            )
            return

        if (fut := self._buffers.get(track_id)) and fut.done():
            # reset the buffer if it's already set
            self._buffers.pop(track_id)
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

                async for chunk in reader:
                    decoder.push(chunk)

                decoder.end_input()

                async for decoded_frame in decoder:
                    buf.frames.append(decoded_frame)
                    duration += decoded_frame.duration
            else:
                # Process raw audio directly through AudioByteStream
                audio_stream = utils.audio.AudioByteStream(sample_rate, num_channels)
                async for chunk in reader:
                    for frame in audio_stream.push(chunk):
                        buf.frames.append(frame)
                        duration += frame.duration

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
        except Exception as e:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_exception(e)
