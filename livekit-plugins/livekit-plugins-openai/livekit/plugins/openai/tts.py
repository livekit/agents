import os
import asyncio
from typing import AsyncIterator
from openai import AsyncOpenAI
from livekit import rtc
import audioread


class TTSPlugin:

    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def close(self):
        pass

    async def generate_speech_from_text(self, text: str) -> AsyncIterator[rtc.AudioFrame]:
        async def iterator():
            yield text

        return self.generate_speech_from_stream(iterator())

    async def generate_speech_from_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[rtc.AudioFrame]:
        def create_directory():
            os.makedirs("/tmp/openai_tts", exist_ok=True)

        await asyncio.get_event_loop().run_in_executor(None, create_directory)

        complete_text = ""
        async for text in text_stream:
            complete_text += text

        response = await self._client.audio.speech.create(model="tts-1", voice="alloy", response_format="mp3", input=complete_text, )
        filepath = "/tmp/openai_tts/output.mp3"
        response.stream_to_file(filepath)
        with audioread.audio_open(filepath) as f:
            for buf in f:
                frame = rtc.AudioFrame(buf, f.samplerate, f.channels, len(buf) // 2)
                yield frame
