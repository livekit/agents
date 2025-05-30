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
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

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
        base_url: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "playai-tts",
        voice: TTSVoices | str = "Arista-PlayAI",
        api_key: NotGivenOr[str] = NOT_GIVEN,
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
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

        self._session = http_session

        if not base_url:
            base_url = DEFAULT_BASE_URL

        groq_api_key = api_key if is_given(api_key) else os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            api_key=groq_api_key,
            base_url=base_url,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self, *, model: NotGivenOr[TTSModels] = NOT_GIVEN, voice: NotGivenOr[TTSVoices] = NOT_GIVEN
    ) -> None:
        """
        Update the TTS options.

        Args:
            model (SpeechModels | str, optional): Model to use. Default is None.
            voice (SpeechVoices | str, optional): Voice to use. Default is None.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        api_url = f"{self._opts.base_url}/audio/speech"
        try:
            async with self._tts._ensure_session().post(
                api_url,
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._opts.model,
                    "voice": self._opts.voice,
                    "input": self._input_text,
                    "response_format": "wav",
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio"):
                    content = await resp.text()
                    raise APIError(message="Groq returned non-audio data", body=content)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
