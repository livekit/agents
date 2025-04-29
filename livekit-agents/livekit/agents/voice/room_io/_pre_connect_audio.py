import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from livekit import rtc

from ..agent import logger, utils

PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"


@dataclass
class PreConnectAudioData:
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    received_fut: asyncio.Future[None]
    _consumed: bool = False

    def read_audio(
        self, *, timeout: float = 2.0, sample_rate: int | None = None
    ) -> AsyncIterator[rtc.AudioFrame]:
        async def _read_audio():
            if self._consumed:
                return
            self._consumed = True

            try:
                await asyncio.wait_for(self.received_fut, timeout)
            except asyncio.TimeoutError as e:
                logger.warning("timeout waiting for pre-connect audio buffer")
                self.received_fut.set_exception(e)
                return

            resampler: rtc.AudioResampler | None = None
            async for frame in self.audio_ch:
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

        return _read_audio()


class PreConnectAudioHandler:
    def __init__(self, room: rtc.Room):
        self._room = room
        self._data = PreConnectAudioData(utils.aio.Chan[rtc.AudioFrame](), asyncio.Future())
        self._main_atask: asyncio.Task | None = None

    def start(self) -> PreConnectAudioData:
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
            # already timeout
            return

        logger.info("start reading audio buffer")
        data_ch = self._data.audio_ch
        self._data.received_fut.set_result(None)
        try:
            sample_rate = int(reader.info.attributes["sampleRate"])
            num_channels = int(reader.info.attributes["channels"])
            logger.info(f"sample rate: {sample_rate}, num channels: {num_channels}")

            duration = 0
            audio_stream = utils.audio.AudioByteStream(sample_rate, num_channels)
            async for chunk in reader:
                for frame in audio_stream.push(chunk):
                    await data_ch.send(frame)
                    duration += frame.duration

            for frame in audio_stream.flush():
                await data_ch.send(frame)
                duration += frame.duration
        finally:
            data_ch.close()
            logger.info(f"audio buffer duration: {duration}")
