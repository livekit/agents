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
from typing import Optional

import aiohttp
from livekit.agents import tts, utils
from livekit.agents.utils import codecs

from .log import logger
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
        model: TTSModels = "tts-1",
        voice: TTSVoices = "alloy",
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
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
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        self._opts = opts
        self._text = text
        self._session = session
        self._decoder = codecs.Mp3StreamDecoder()
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    async def _run(self):
        try:
            async with self._session.post(
                OPENAI_ENPOINT,
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                json={
                    "input": self._text,
                    "model": self._opts.model,
                    "voice": self._opts.voice,
                    "response_format": "mp3",
                },
            ) as resp:
                async for data, _ in resp.content.iter_chunks():
                    frames = self._decoder.decode_chunk(data)
                    for frame in frames:
                        self._queue.put_nowait(
                            tts.SynthesizedAudio(text=self._text, data=frame)
                        )

        except Exception:
            logger.exception("openai tts main task failed in chunked stream")
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
