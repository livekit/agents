import asyncio
import contextlib
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Callable

from livekit import rtc

from ..agent import logger, utils

PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"


@dataclass
class _PreConnectAudioBuffer:
    timestamp: float
    frames: list[rtc.AudioFrame] = field(default_factory=list)


_WaitPreConnectAudio = Callable[[str], Coroutine[Any, Any, list[rtc.AudioFrame]]]


class PreConnectAudioHandler:
    def __init__(self, room: rtc.Room, *, timeout: float, max_delta_s: float = 1.0):
        self._room = room
        self._timeout = timeout
        self._max_delta_s = max_delta_s

        self._buffers: dict[str, asyncio.Future[_PreConnectAudioBuffer]] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

        def _handler(reader: rtc.ByteStreamReader, participant_id: str):
            task = asyncio.create_task(self._read_audio_task(reader, participant_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, _handler)

    async def aclose(self):
        self._room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM)
        await utils.aio.cancel_and_wait(*self._tasks)

    async def wait_for_data(self, participant_identity: str) -> list[rtc.AudioFrame]:
        async with self._lock:
            self._buffers.setdefault(participant_identity, asyncio.Future())
            fut = self._buffers[participant_identity]

        try:
            if fut.done():
                buf = fut.result()
                if (delta := time.time() - buf.timestamp) > self._max_delta_s:
                    logger.warning(
                        "pre-connect audio buffer is too old",
                        extra={"participant": participant_identity, "delta_time": delta},
                    )
                    return []
                return buf.frames

            buf = await asyncio.wait_for(fut, self._timeout)
            return buf.frames
        finally:
            async with self._lock:
                self._buffers.pop(participant_identity)

    @utils.log_exceptions(logger=logger)
    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str):
        async with self._lock:
            if (fut := self._buffers.get(participant_id)) and fut.done():
                # reset the buffer if it's already set
                self._buffers.pop(participant_id)
            self._buffers.setdefault(participant_id, asyncio.Future())
            fut = self._buffers[participant_id]

        buf = _PreConnectAudioBuffer(timestamp=time.time())
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
