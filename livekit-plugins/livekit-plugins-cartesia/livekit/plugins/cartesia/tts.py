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
from livekit import rtc
from livekit.agents import tts, utils

from .log import logger
from .models import TTSDefaultVoiceId, TTSEncoding, TTSModels

API_AUTH_HEADER = "X-API-Key"
API_VERSION_HEADER = "Cartesia-Version"
API_VERSION = "2024-06-10"


@dataclass
class _TTSOptions:
    model: TTSModels
    encoding: TTSEncoding
    sample_rate: int
    voice: str | list[float]
    api_key: str
    language: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels = "sonic-english",
        language: str = "en",
        encoding: TTSEncoding = "pcm_s16le",
        voice: str | list[float] = TTSDefaultVoiceId,
        sample_rate: int = 24000,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError("CARTESIA_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            api_key=api_key,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        self._opts = opts
        self._text = text
        self._session = session
        self._main_task: asyncio.Task[None] | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    @utils.log_exceptions(logger=logger)
    async def _run(self):
        segment_id = utils.nanoid()
        voice = {}
        if isinstance(self._opts.voice, str):
            voice["mode"] = "id"
            voice["id"] = self._opts.voice
        else:
            voice["mode"] = "embedding"
            voice["embedding"] = self._opts.voice

        try:
            async with self._session.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    API_AUTH_HEADER: f"{self._opts.api_key}",
                    API_VERSION_HEADER: API_VERSION,
                },
                json={
                    "model_id": self._opts.model,
                    "transcript": self._text,
                    "voice": voice,
                    "output_format": {
                        "container": "raw",
                        "encoding": self._opts.encoding,
                        "sample_rate": self._opts.sample_rate,
                    },
                    "language": self._opts.language,
                },
            ) as resp:
                bytes_per_frame = (self._opts.sample_rate // 100) * 2
                buf = bytearray()

                async for data, _ in resp.content.iter_chunks():
                    buf.extend(data)

                    while len(buf) >= bytes_per_frame:
                        frame_data = buf[:bytes_per_frame]
                        buf = buf[bytes_per_frame:]

                        self._queue.put_nowait(
                            tts.SynthesizedAudio(
                                segment_id=segment_id,
                                frame=rtc.AudioFrame(
                                    data=frame_data,
                                    sample_rate=self._opts.sample_rate,
                                    num_channels=1,
                                    samples_per_channel=len(frame_data) // 2,
                                ),
                            )
                        )

                # send any remaining data
                if len(buf) > 0:
                    self._queue.put_nowait(
                        tts.SynthesizedAudio(
                            segment_id=segment_id,
                            frame=rtc.AudioFrame(
                                data=buf,
                                sample_rate=self._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(buf) // 2,
                            ),
                        )
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
