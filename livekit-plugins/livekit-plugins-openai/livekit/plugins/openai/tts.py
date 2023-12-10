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
import tempfile
from typing import AsyncIterator
from openai import AsyncOpenAI
from livekit import rtc
import audioread


class TTSPlugin:
    """Text-to-speech plugin using OpenAI's API
    """

    def __init__(self, model = "tts-1", voice = "alloy"):
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model
        self._voice = voice

    async def close(self):
        pass

    async def generate_speech_from_text(self, text: str) -> AsyncIterator[rtc.AudioFrame]:
        """Generate a stream of speech from text

        Args:
            text (str): Text to generate speech from

        Returns:
            AsyncIterator[rtc.AudioFrame]: Stream of 24000hz, 1 channel audio frames
        """
        async def iterator():
            yield text

        return self.generate_speech_from_stream(iterator())

    async def generate_speech_from_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[rtc.AudioFrame]:
        """Generate a stream of speech from a stream of text

        Args:
            text_stream (AsyncIterator[str]): Stream of text to generate speech from

        Returns:
            AsyncIterator[rtc.AudioFrame]: Stream of 24000hz, 1 channel audio frames
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            complete_text = ""
            async for text in text_stream:
                complete_text += text

            response = await self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                response_format="mp3",
                input=complete_text,
            )
            filepath = os.path.join(tmpdir, "output.mp3")
            response.stream_to_file(filepath)
            with audioread.audio_open(filepath) as f:
                for buf in f:
                    frame = rtc.AudioFrame(
                        buf, f.samplerate, f.channels, len(buf) // 2)
                    yield frame
