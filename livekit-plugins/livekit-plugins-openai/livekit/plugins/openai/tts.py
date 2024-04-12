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
from typing import AsyncIterable, Optional

import aiohttp
from livekit.agents import codecs, tts

from .models import TTSModels, TTSVoices

OPENAI_TTS_SAMPLE_RATE = 24000
OPENAI_TTS_CHANNELS = 1
OPENAI_ENPOINT = "https://api.openai.com/v1/audio/speech"


class TTS(tts.TTS):
    def __init__(
        self, model: TTSModels, voice: TTSVoices, api_key: Optional[str] = None
    ) -> None:
        super().__init__(streaming_supported=False)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        # TODO: we want to reuse aiohttp sessions
        # for improved latency but doing so doesn't
        # give us a clean way to close the session.
        # Perhaps we introduce a close method to TTS?
        # We also probalby want to send a warmup HEAD
        # request after we create this
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        )

        self._model = model
        self._voice = voice

    def synthesize(
        self,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        decoder = codecs.Mp3StreamDecoder()

        async def generator():
            async with self._session.post(
                OPENAI_ENPOINT,
                json={
                    "input": text,
                    "model": self._model,
                    "voice": self._voice,
                    "response_format": "mp3",
                },
            ) as resp:
                async for data in resp.content.iter_chunked(4096):
                    frames = decoder.decode_chunk(data)
                    for frame in frames:
                        yield tts.SynthesizedAudio(text=text, data=frame)

        return generator()
