import logging
import asyncio
import os
import io
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional

from openai import AsyncOpenAI
from livekit import rtc
from livekit.plugins import core
import numpy as np


WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


class WhisperAPITranscriber(core.STTPlugin):

    def __init__(self):
        super().__init__(process=self._process, close=self._close)
        self._model = None
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._result_iterator = core.AsyncQueueIterator(asyncio.Queue[core.STTPluginResult]())
        self._task = None
        self._frame_streams = None

    def _process(self, frame_streams: AsyncIterable[List[rtc.AudioFrame]]) -> AsyncIterable[AsyncIterable[core.STTPluginResult]]:
        self._frame_streams = frame_streams
        self._task = asyncio.create_task(self._async_process(frame_streams))
        return self._result_iterator

    async def _close(self):
        pass

    async def _async_process(self, frame_streams: AsyncIterable[List[rtc.AudioFrame]]) -> AsyncIterable[core.STTPluginResult]:
        async for frame_stream in frame_streams:
            if len(frame_stream) == 0:
                continue
            sample_rate = frame_stream[0].sample_rate
            channels = frame_stream[0].num_channels
            full_buffer = bytearray()
            for frame in frame_stream:
                full_buffer += frame.data

            try:
                bytes_io = io.BytesIO(full_buffer)
                with wave.open(bytes_io, mode="wb") as wave_file:
                    wave_file.setnchannels(channels)
                    wave_file.setsampwidth(2)  # int16
                    wave_file.setframerate(sample_rate)
                    wave_file.writeframes(full_buffer)

                response = await self._client.audio.transcriptions.create(file=("input.wav", bytes_io), model="whisper-1", response_format="text")
                result = core.STTPluginResult(type=core.STTPluginResultType.DELTA_RESULT, text=response)
                await self._result_iterator.put(core.AsyncIteratorList([result]))
            except Exception as e:
                logging.error("Error transcribing audio: %s", e)
