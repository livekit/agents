# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import asyncio
import os
import io
import wave
from typing import List
from openai import AsyncOpenAI
from livekit import rtc


WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


class WhisperAPITranscriber:
    """Plugin that uses OpenAI's Whisper API to generate text from audio
    """

    def __init__(self):
        self._model = None
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._task = None

    async def close(self):
        pass

    async def transcribe_frames(self, frames: List[rtc.AudioFrame]) -> str:
        """Generate text from a list of audio frames

        Args:
            frames (List[rtc.AudioFrame]): List of audio frames to generate text from

        Returns:
            str: Text transcribed from audio frames
        """
        if len(frames) == 0:
            return ""

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
            return response
        except Exception as e:
            logging.error("Error transcribing audio: %s", e)
            return ""
