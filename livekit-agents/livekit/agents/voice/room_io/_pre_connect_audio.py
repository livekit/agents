import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from livekit import rtc

from ..agent import logger, utils

PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"


@dataclass
class PreConnectAudioData:
    received_fut: asyncio.Future[None]
    frames: list[rtc.AudioFrame] = field(default_factory=list)

    def read_audio(
        self, *, timeout: float = 2.0, sample_rate: int | None = None
    ) -> AsyncIterator[rtc.AudioFrame]:
        async def _read_audio():
            if self.received_fut.done() and not self.frames:
                return

            try:
                await asyncio.wait_for(self.received_fut, timeout)
            except asyncio.TimeoutError:
                logger.warning("timeout waiting for pre-connect audio buffer")
                return
            except Exception as e:
                logger.error("error reading pre-connect audio buffer", extra={"error": e})
                return

            resampler: rtc.AudioResampler | None = None
            for frame in self.frames:
                if not resampler and sample_rate is not None and frame.sample_rate != sample_rate:
                    resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=sample_rate,
                    )
                if resampler:
                    for f in resampler.push(frame):
                        yield f
                else:
                    yield frame
            if resampler:
                for f in resampler.flush():
                    yield f
            self.frames.clear()

        return _read_audio()


class PreConnectAudioHandler:
    def __init__(self, room: rtc.Room):
        self._room = room
        self._data = PreConnectAudioData(asyncio.Future())
        self._main_atask: asyncio.Task | None = None

    def register(self) -> PreConnectAudioData:
        def _on_audio_buffer(reader: rtc.ByteStreamReader, participant_id: str):
            if self._main_atask is not None:
                logger.warning("received audio buffer multiple times, ignoring")
                return
            self._main_atask = asyncio.create_task(self._read_audio_task(reader, participant_id))

        self._room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, _on_audio_buffer)
        self._data.received_fut.add_done_callback(
            lambda _: self._room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM)
        )
        return self._data

    @utils.log_exceptions(logger=logger)
    async def _read_audio_task(self, reader: rtc.ByteStreamReader, participant_id: str):
        if self._data.received_fut.done():
            return

        try:
            sample_rate = int(reader.info.attributes["sampleRate"])
            num_channels = int(reader.info.attributes["channels"])
            logger.debug(
                "received pre-connect audio",
                extra={"sample_rate": sample_rate, "num_channels": num_channels},
            )

            audio_stream = utils.audio.AudioByteStream(sample_rate, num_channels)
            async for chunk in reader:
                for frame in audio_stream.push(chunk):
                    self._data.frames.append(frame)

            for frame in audio_stream.flush():
                self._data.frames.append(frame)

            self._data.received_fut.set_result(None)
        except Exception as e:
            self._data.received_fut.set_exception(e)
