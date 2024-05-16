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

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass

import aiohttp
from livekit.agents import codecs, tts

from .models import TTSModels, TTSVoices

OPENAI_TTS_SAMPLE_RATE = 24000
OPENAI_TTS_CHANNELS = 1
OPENAI_ENPOINT = "https://api.openai.com/v1/audio/speech"


@dataclass
class _TTSOptions:
    model: TTSModels
    voice: TTSVoices
    api_key: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels,
        voice: TTSVoices,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            streaming_supported=False,
            sample_rate=OPENAI_TTS_SAMPLE_RATE,
            num_channels=OPENAI_TTS_CHANNELS,
        )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self._opts = _TTSOptions(model=model, voice=voice, api_key=api_key)

        # TODO: we want to reuse aiohttp sessions
        # for improved latency but doing so doesn't
        # give us a clean way to close the session.
        # Perhaps we introduce a close method to TTS?
        # We also probalby want to send a warmup HEAD
        # request after we create this
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        )

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._session)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        self._opts = opts
        self._text = text
        self._session = session
        self._decoder = codecs.Mp3StreamDecoder()
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[tts.SynthesizedAudio | None]()

    async def _run(self):
        try:
            async with self._session.post(
                OPENAI_ENPOINT,
                json={
                    "input": self._text,
                    "model": self._opts.model,
                    "voice": self._opts.voice,
                    "response_format": "mp3",
                },
            ) as resp:
                async for data in resp.content.iter_chunked(4096):
                    frames = self._decoder.decode_chunk(data)
                    for frame in frames:
                        self._queue.put_nowait(
                            tts.SynthesizedAudio(text="", data=frame)
                        )

        finally:
            self._queue.put_nowait(None)

    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._main_task:
            self._main_task = asyncio.create_task(self._run())

        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration

        return frame

    async def aclose(self) -> None:
        if not self._main_task:
            return

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task
