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
                frame = rtc.AudioFrame(
                    buf, f.samplerate, f.channels, len(buf) // 2)
                yield frame
