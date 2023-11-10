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
        self._model = None
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._response_iterator = core.AsyncQueueIterator(asyncio.Queue[rtc.AudioFrame]())
        super().__init__(process=self.process)

    def process(self, text_streams: AsyncIterator[AsyncIterator[str]]) -> AsyncIterator[AsyncIterator[rtc.AudioFrame]]:
        asyncio.create_task(self._async_process(text_streams))
        return self._response_iterator

    async def _async_process(self, text_streams: AsyncIterator[AsyncIterator[str]]) -> AsyncIterator[AsyncIterator[rtc.AudioFrame]]:
        loop = asyncio.get_event_loop()

        def create_directory():
            os.makedirs("/tmp/openai_tts", exist_ok=True)

        await loop.run_in_executor(None, create_directory)

        async for text_stream in text_streams:
            complete_text = ""
            async for text in text_stream:
                complete_text += text
            response = await self._client.audio.speech.create(model="tts-1", voice="alloy", response_format="mp3", input=complete_text, )
            filepath = "/tmp/openai_tts/output.mp3"
            response.stream_to_file(filepath)
            audio_stream = core.AsyncQueueIterator(asyncio.Queue[rtc.AudioFrame]())
            with audioread.audio_open(filepath) as f:
                for buf in f:
                    frame = rtc.AudioFrame(buf, f.samplerate, f.channels, len(buf) // 2)
                    await audio_stream.put(frame)

                await audio_stream.aclose()

            await self._response_iterator.put(audio_stream)
