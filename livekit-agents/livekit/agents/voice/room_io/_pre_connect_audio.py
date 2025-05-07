import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from livekit import rtc

from ..agent import logger, utils

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
        self._tasks: set[asyncio.Task] = set()

    def register(self):
        def _handler(reader: rtc.ByteStreamReader, participant_id: str):
            task = asyncio.create_task(self._read_audio_task(reader, participant_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            def _on_timeout():
                logger.warning(
                    "pre-connect audio received but not completed in time",
                    extra={"participant": participant_id},
                )
                if not task.done():
                    task.cancel()

            timeout_handle = asyncio.get_event_loop().call_later(self._timeout, _on_timeout)
            task.add_done_callback(lambda _: timeout_handle.cancel())

        self._room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, _handler)

    async def aclose(self):
        self._room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM)
        await utils.aio.cancel_and_wait(*self._tasks)

    async def wait_for_data(self, track_id: str) -> list[rtc.AudioFrame]:
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
    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str):
        if not (track_id := reader.info.attributes.get("trackId")):
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
            sample_rate = int(reader.info.attributes["sampleRate"])
            num_channels = int(reader.info.attributes["channels"])

            duration = 0
            audio_stream = utils.audio.AudioByteStream(sample_rate, num_channels)
            async for chunk in reader:
                for frame in audio_stream.push(chunk):
                    buf.frames.append(frame)
                    duration += frame.duration

            for frame in audio_stream.flush():
                buf.frames.append(frame)
                duration += frame.duration

            logger.debug(
                "pre-connect audio received",
                extra={"duration": duration, "track_id": track_id, "participant": participant_id},
            )

            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(buf)
        except Exception as e:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_exception(e)
