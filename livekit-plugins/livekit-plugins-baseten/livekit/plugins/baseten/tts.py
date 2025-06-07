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
import ssl
from dataclasses import dataclass, replace

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


@dataclass
class _TTSOptions:
    model: str
    language: str
    voice: str
    temperature: float


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model: str = "Orpheus",
        voice: str = "tara",
        language: str = "en",
        temperature: float = 0.6,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initialize the Baseten TTS.

        Args:
            api_key (str): Baseten API key.
            model (TTSModel): TTS model, defaults to "Orpheus".
            voice (str): Speaker voice.
            language (str): language, defaults to "english".
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("BASETEN_API_KEY")

        if not api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )

        if not model_endpoint:
            raise ValueError(
                "The model endpoint is required, you can find it in the Baseten dashboard"
            )

        self._api_key = api_key
        self._model_endpoint = model_endpoint

        self._opts = _TTSOptions(
            voice=voice, model=model, language=language, temperature=temperature
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(temperature):
            self._opts.temperature = temperature

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            api_key=self._api_key,
            input_text=text,
            model_endpoint=self._model_endpoint,
            conn_options=conn_options
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
    self,
    *,
    tts: TTS,
    api_key: str,
    model_endpoint: str,
    input_text: str,
    conn_options: APIConnectOptions,
) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._tts = tts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter):
        try:
            async with self._tts._ensure_session().post(
                self._model_endpoint,
                headers = {
                    "Authorization": f"Api-Key {self._api_key}",
                },
                json={
                    "prompt": self._input_text,
                    "voice": self._opts.voice,
                    "temperature": self._opts.temperature
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
                ssl=ssl_context,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=24000,
                    num_channels=1,
                    mime_type="audio/pcm", # TODO: This needs to change to audio/wav I think.
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
