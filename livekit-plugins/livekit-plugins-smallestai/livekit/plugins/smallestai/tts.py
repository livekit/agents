# Copyright 2025 LiveKit, Inc.
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
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    Language,
    create_api_error_from_http,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSEncoding, TTSModels

NUM_CHANNELS = 1
SMALLEST_BASE_URL = "https://waves-api.smallest.ai/api/v1"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    api_key: str
    voice_id: str
    sample_rate: int
    speed: float
    consistency: float
    similarity: float
    enhancement: float
    language: Language
    output_format: TTSEncoding | str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "lightning-large",
        voice_id: str = "irisha",
        sample_rate: int = 24000,
        speed: float = 1.0,
        consistency: float = 0.5,
        similarity: float = 0,
        enhancement: float = 1,
        language: str = "en",
        output_format: TTSEncoding | str = "pcm",
        base_url: str = SMALLEST_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of smallest.ai Waves TTS.
        Args:
            api_key: Your Smallest AI API key.
            model: The TTS model to use (e.g., "lightning", "lightning-large", "lightning-v2").
            voice_id: The voice ID to use for synthesis.
            sample_rate: Sample rate for the audio output.
            speed: Speed of the speech synthesis.
            consistency: Consistency of the speech synthesis.
            similarity: Similarity of the speech synthesis.
            enhancement: Enhancement level for the speech synthesis.
            language: Language of the text to be synthesized.
            output_format: Output format of the audio.
            base_url: Base URL for the Smallest AI API.
            http_session: An existing aiohttp ClientSession to use.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest.ai API key is required, either as argument or set"
                " SMALLEST_API_KEY environment variable"
            )

        if (consistency or similarity or enhancement) and model == "lightning":
            logger.warning(
                "consistency, similarity, and enhancement are only supported for model 'lightning-large' and 'lightning-v2'. "
            )

        self._opts = _TTSOptions(
            model=model,
            api_key=api_key,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            consistency=consistency,
            similarity=similarity,
            enhancement=enhancement,
            language=Language(language),
            output_format=output_format,
            base_url=base_url,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SmallestAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        consistency: NotGivenOr[float] = NOT_GIVEN,
        similarity: NotGivenOr[float] = NOT_GIVEN,
        enhancement: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[TTSEncoding | str] = NOT_GIVEN,
    ) -> None:
        """Update TTS options."""
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(consistency):
            self._opts.consistency = consistency
        if is_given(similarity):
            self._opts.similarity = similarity
        if is_given(enhancement):
            self._opts.enhancement = enhancement
        if is_given(language):
            self._opts.language = Language(language)
        if is_given(output_format):
            self._opts.output_format = output_format

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the Waves API endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the chunked synthesis process."""
        try:
            data = _to_smallest_options(self._opts)
            data["text"] = self._input_text

            url = f"{SMALLEST_BASE_URL}/{self._opts.model}/get_speech_long_text"
            if self._opts.model == "lightning-v2":
                url = f"{SMALLEST_BASE_URL}/{self._opts.model}/get_speech"

            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
            }
            async with self._tts._ensure_session().post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.output_format}",
                )

                async for chunk, _ in resp.content.iter_chunks():
                    output_emitter.push(chunk)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from None
        except Exception as e:
            raise APIConnectionError() from e


def _to_smallest_options(opts: _TTSOptions) -> dict[str, Any]:
    base_keys = ["voice_id", "sample_rate", "speed", "language", "output_format"]
    extra_keys = ["consistency", "similarity", "enhancement"]

    keys = base_keys if opts.model == "lightning" else base_keys + extra_keys
    result = {key: getattr(opts, key) for key in keys}
    if "language" in result and isinstance(result["language"], Language):
        result["language"] = result["language"].language
    return result
