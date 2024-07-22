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
from typing import AsyncContextManager, Optional

from livekit.agents import tts
from livekit.agents.utils import codecs, nanoid

import openai

from .log import logger
from .models import TTSModels, TTSVoices
from .utils import AsyncAzureADTokenProvider, get_base_url

OPENAI_TTS_SAMPLE_RATE = 24000
OPENAI_TTS_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModels
    voice: TTSVoices
    endpoint: str
    speed: float


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels = "tts-1",
        voice: TTSVoices = "alloy",
        speed: float = 1.0,
        base_url: str | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=OPENAI_TTS_SAMPLE_RATE,
            num_channels=OPENAI_TTS_CHANNELS,
        )

        self._client = client or openai.AsyncClient(base_url=get_base_url(base_url))

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            endpoint=os.path.join(get_base_url(base_url), "audio/speech"),
            speed=speed,
        )

    @staticmethod
    def create_azure_client(
        *,
        model: TTSModels = "tts-1",
        voice: TTSVoices = "alloy",
        speed: float = 1.0,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
    ) -> TTS:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """

        azure_client = openai.AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
        )  # type: ignore

        return TTS(model=model, voice=voice, speed=speed, client=azure_client)

    def synthesize(self, text: str) -> "ChunkedStream":
        stream = self._client.audio.speech.with_streaming_response.create(
            input=text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format="mp3",
            speed=self._opts.speed,
        )

        return ChunkedStream(stream, text, self._opts)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        oai_stream: AsyncContextManager[openai.AsyncAPIResponse[bytes]],
        text: str,
        opts: _TTSOptions,
    ) -> None:
        self._opts, self._text = opts, text
        self._oai_stream = oai_stream
        self._decoder = codecs.Mp3StreamDecoder()
        self._main_task: asyncio.Task[None] | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    async def _run(self):
        try:
            segment_id = nanoid()
            async with self._oai_stream as stream:
                async for data in stream.iter_bytes(4096):
                    frames = self._decoder.decode_chunk(data)
                    for frame in frames:
                        self._queue.put_nowait(
                            tts.SynthesizedAudio(segment_id=segment_id, frame=frame)
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
