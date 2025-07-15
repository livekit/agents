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
from dataclasses import dataclass, replace
from typing import Literal, Union

import httpx

import openai
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import aio, is_given

from .models import TTSModels, TTSVoices
from .utils import AsyncAzureADTokenProvider

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "ash"

RESPONSE_FORMATS = Union[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"], str]


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    speed: float
    instructions: str | None
    response_format: RESPONSE_FORMATS


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: TTSVoices | str = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        response_format: NotGivenOr[RESPONSE_FORMATS] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of OpenAI TTS.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            speed=speed,
            instructions=instructions if is_given(instructions) else None,
            response_format=response_format if is_given(response_format) else "mp3",
        )

        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
                ),
            ),
        )

        self._prewarm_task: asyncio.Task | None = None

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(instructions):
            self._opts.instructions = instructions

    @staticmethod
    def with_azure(
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: TTSVoices | str = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        response_format: NotGivenOr[RESPONSE_FORMATS] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> TTS:
        """
        Create a new instance of Azure OpenAI TTS.

        This automatically infers the following arguments from their corresponding environment
        variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """

        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )  # type: ignore

        return TTS(
            model=model,
            voice=voice,
            speed=speed,
            instructions=instructions,
            client=azure_client,
            response_format=response_format,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        async def _prewarm() -> None:
            try:
                await self._client.get("/", cast_to=str)
            except Exception:
                pass

        self._prewarm_task = asyncio.create_task(_prewarm())

    async def aclose(self) -> None:
        if self._prewarm_task:
            await aio.cancel_and_wait(self._prewarm_task)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        oai_stream = self._tts._client.audio.speech.with_streaming_response.create(
            input=self.input_text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format=self._opts.response_format,  # type: ignore
            speed=self._opts.speed,
            instructions=self._opts.instructions or openai.NOT_GIVEN,
            timeout=httpx.Timeout(30, connect=self._conn_options.timeout),
        )

        try:
            async with oai_stream as stream:
                output_emitter.initialize(
                    request_id=stream.request_id or "",
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.response_format}",
                )

                async for data in stream.iter_bytes():
                    output_emitter.push(data)

            output_emitter.flush()

        except openai.APITimeoutError:
            raise APITimeoutError() from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=e.request_id, body=e.body
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
