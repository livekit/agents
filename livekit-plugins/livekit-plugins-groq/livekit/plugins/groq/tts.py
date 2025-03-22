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
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import TTSModels, TTSVoices

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
SAMPLE_RATE = 48000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: TTSModels | str = "playai-tts",
        voice: TTSVoices | str = "Eileen-PlayAI",
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Groq TTS.

        if `api_key` is not provided, it will be read from the ``GROQ_API_KEY``
        environmental variable.

        Args:
            model (SpeechModels | str, optional): Model to use. Default is "playai-tts".
            voice (SpeechVoices | str, optional): Voice to use. Default is "Autumn-PlayAI".
            api_key (str | None, optional): API key to use. Default is None.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

        self._session = http_session

        if not base_url:
            base_url = DEFAULT_BASE_URL

        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set")

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        model: TTSModels | None = None,
        voice: TTSVoices | None = None,
    ) -> None:
        """
        Update the TTS options.

        Args:
            model (SpeechModels | str, optional): Model to use. Default is None.
            voice (SpeechVoices | str, optional): Voice to use. Default is None.
        """
        if model:
            self._opts.model = model
        if voice:
            self._opts.voice = voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
        segment_id: str | None = None,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
            segment_id=segment_id,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: Optional[APIConnectOptions] = None,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        segment_id: str | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._segment_id = segment_id

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._opts.model,
            "voice": self._opts.voice,
            "input": self._input_text,
            "response_format": "wav",
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        decode_task: Optional[asyncio.Task] = None
        api_url = f"{self._opts.base_url}/audio/speech"
        try:
            async with self._session.post(
                api_url, headers=headers, json=payload
            ) as response:
                if not response.content_type.startswith("audio"):
                    content = await response.text()
                    logger.error("Groq returned non-audio data: %s", content)
                    return

                async def _decode_loop():
                    try:
                        async for bytes_data, _ in response.content.iter_chunks():
                            decoder.push(bytes_data)
                    finally:
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    segment_id=self._segment_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()
