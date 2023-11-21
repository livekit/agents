import io
import os
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from openai import AsyncOpenAI
import torch

import whisper
from livekit import rtc
from livekit.plugins import core
import numpy as np
import audioread


class TTSPlugin(core.TTSPlugin):

    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def close(self):
        pass

    async def process(self, text_stream: AsyncIterator[str]) -> AsyncIterator[rtc.AudioFrame]:
        res = core.PluginIterator[rtc.AudioFrame]()
        loop = asyncio.get_event_loop()

        def create_directory():
            os.makedirs("/tmp/openai_tts", exist_ok=True)

        await loop.run_in_executor(None, create_directory)

        async def generate_result():
            complete_text = ""
            async for text in text_stream:
                complete_text += text

            response = await self._client.audio.speech.create(model="tts-1", voice="alloy", response_format="mp3", input=complete_text, )
            filepath = "/tmp/openai_tts/output.mp3"
            response.stream_to_file(filepath)
            with audioread.audio_open(filepath) as f:
                for buf in f:
                    frame = rtc.AudioFrame(
                        buf, f.samplerate, f.channels, len(buf) // 2)
                    await res.put(frame)

                await res.aclose()

        asyncio.create_task(generate_result())
        return res
