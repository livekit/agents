import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from livekit import rtc

from ..agent import logger, utils

PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"


@dataclass
class _PreConnectBuffer:
    timestamp: float
    frames: list[rtc.AudioFrame] = field(default_factory=list)


@dataclass
class PreConnectAudioData:
    buffers: dict[str, asyncio.Future[_PreConnectBuffer]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def wait_for_data(
        self, *, timeout: float, participant_identity: str
    ) -> list[rtc.AudioFrame]:
        async with self._lock:
            self.buffers.setdefault(participant_identity, asyncio.Future())
            fut = self.buffers[participant_identity]

        try:
            if fut.done():
                buf = fut.result()
                if (delta := time.time() - buf.timestamp) > 1.0:
                    logger.warning(
                        "pre-connect audio buffer is too old",
                        extra={"participant": participant_identity, "delta_time": delta},
                    )
                    return []
                return buf.frames

            buf = await asyncio.wait_for(fut, timeout)
            return buf.frames
        finally:
            async with self._lock:
                self.buffers.pop(participant_identity)


class PreConnectAudioHandler:
    def __init__(self, room: rtc.Room):
        self._room = room
        self._data = PreConnectAudioData()
        self._tasks: set[asyncio.Task] = set()

    def register(self) -> PreConnectAudioData:
        def _on_audio_buffer(reader: rtc.ByteStreamReader, participant_id: str):
            task = asyncio.create_task(self._read_audio_task(reader, participant_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, _on_audio_buffer)
        return self._data

    async def aclose(self):
        self._room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM)
        await utils.aio.cancel_and_wait(*self._tasks)

    @utils.log_exceptions(logger=logger)
    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str):
        async with self._data._lock:
            if (fut := self._data.buffers.get(participant_id)) and fut.done():
                # reset the buffer if it's already set
                self._data.buffers.pop(participant_id)
            self._data.buffers.setdefault(participant_id, asyncio.Future())
            fut = self._data.buffers[participant_id]

        buf = _PreConnectBuffer(timestamp=time.time())
        try:
            sample_rate = int(reader.info.attributes["sampleRate"])
            num_channels = int(reader.info.attributes["channels"])
            logger.debug(
                "pre-connect audio connected",
                extra={
                    "sample_rate": sample_rate,
                    "num_channels": num_channels,
                    "participant": participant_id,
                },
            )

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
                extra={"duration": duration, "participant": participant_id},
            )

            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_result(buf)
        except Exception as e:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_exception(e)
