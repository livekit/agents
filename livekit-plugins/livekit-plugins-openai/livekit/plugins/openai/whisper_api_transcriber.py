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
        self._model = None
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._task = None

    async def close(self):
        pass

    async def transcribe_frames(self, frames: List[rtc.AudioFrame]) -> AsyncIterable[core.STTPluginResult]:
        if len(frames) == 0:
            return

        sample_rate = frames[0].sample_rate
        channels = frames[0].num_channels
        full_buffer = bytearray()
        for frame in frames:
            full_buffer += frame.data

        try:
            bytes_io = io.BytesIO(full_buffer)
            with wave.open(bytes_io, mode="wb") as wave_file:
                wave_file.setnchannels(channels)
                wave_file.setsampwidth(2)  # int16
                wave_file.setframerate(sample_rate)
                wave_file.writeframes(full_buffer)

            response = await asyncio.wait_for(self._client.audio.transcriptions.create(file=("input.wav", bytes_io), model="whisper-1", response_format="text"), 10)
            result = core.STTPluginResult(
                type=core.STTPluginResultType.DELTA_RESULT, text=response)
            yield result
        except Exception as e:
            logging.error("Error transcribing audio: %s", e)

