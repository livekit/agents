# Copyright 202 LiveKit, Inc.
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

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
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

from .langs import TTSLangs
from .log import logger
from .models import ArcanaVoices, TTSModels

# arcana can take as long as 80% of the total duration of the audio it's synthesizing.
ARCANA_MODEL_TIMEOUT = 60 * 4
MISTV2_MODEL_TIMEOUT = 30
RIME_BASE_URL = "https://users.rime.ai/v1/rime-tts"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    arcana_options: _ArcanaOptions | None = None
    mistv2_options: _Mistv2Options | None = None


@dataclass
class _ArcanaOptions:
    repetition_penalty: NotGivenOr[float] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    top_p: NotGivenOr[float] = NOT_GIVEN
    max_tokens: NotGivenOr[int] = NOT_GIVEN


@dataclass
class _Mistv2Options:
    lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    sample_rate: NotGivenOr[int] = NOT_GIVEN
    speed_alpha: NotGivenOr[float] = NOT_GIVEN
    reduce_latency: NotGivenOr[bool] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN


NUM_CHANNELS = 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str = RIME_BASE_URL,
        model: TTSModels | str = "arcana",
        speaker: NotGivenOr[ArcanaVoices | str] = NOT_GIVEN,
        # Arcana options
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        # Mistv2 options
        lang: TTSLangs | str = "eng",
        sample_rate: int = 22050,
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = api_key if is_given(api_key) else os.environ.get("RIME_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Rime API key is required, either as argument or set RIME_API_KEY environmental variable"  # noqa: E501
            )

        if not is_given(speaker):
            if model == "mistv2":
                speaker = "cove"
            else:
                speaker = "astra"

        self._opts = _TTSOptions(
            model=model,
            speaker=speaker,
        )
        if model == "arcana":
            self._opts.arcana_options = _ArcanaOptions(
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        elif model == "mistv2":
            self._opts.mistv2_options = _Mistv2Options(
                lang=lang,
                sample_rate=sample_rate,
                speed_alpha=speed_alpha,
                reduce_latency=reduce_latency,
                pause_between_brackets=pause_between_brackets,
                phonemize_between_brackets=phonemize_between_brackets,
            )
        self._session = http_session
        self._base_url = base_url

        self._total_timeout = ARCANA_MODEL_TIMEOUT if model == "arcana" else MISTV2_MODEL_TIMEOUT

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        segment_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
            segment_id=segment_id if is_given(segment_id) else None,
            api_key=self._api_key,
            total_timout=self._total_timeout,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(speaker):
            self._opts.speaker = speaker


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        api_key: str,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
        segment_id: NotGivenOr[str] = NOT_GIVEN,
        total_timout: int = MISTV2_MODEL_TIMEOUT,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._segment_id = segment_id if is_given(segment_id) else utils.shortuuid()
        self._api_key = api_key
        self._total_timeout = total_timout

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        payload = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.model,
        }
        format = "mp3"
        if self._opts.model == "arcana":
            arcana_opts = self._opts.arcana_options
            if is_given(arcana_opts.repetition_penalty):
                payload["repetition_penalty"] = arcana_opts.repetition_penalty
            if is_given(arcana_opts.temperature):
                payload["temperature"] = arcana_opts.temperature
            if is_given(arcana_opts.top_p):
                payload["top_p"] = arcana_opts.top_p
            if is_given(arcana_opts.max_tokens):
                payload["max_tokens"] = arcana_opts.max_tokens
            format = "wav"
        elif self._opts.model == "mistv2":
            mistv2_opts = self._opts.mistv2_options
            if is_given(mistv2_opts.lang):
                payload["lang"] = mistv2_opts.lang
            if is_given(mistv2_opts.sample_rate):
                payload["samplingRate"] = mistv2_opts.sample_rate
            if is_given(mistv2_opts.speed_alpha):
                payload["speedAlpha"] = mistv2_opts.speed_alpha
            if is_given(mistv2_opts.reduce_latency):
                payload["reduceLatency"] = mistv2_opts.reduce_latency
            if is_given(mistv2_opts.pause_between_brackets):
                payload["pauseBetweenBrackets"] = mistv2_opts.pause_between_brackets
            if is_given(mistv2_opts.phonemize_between_brackets):
                payload["phonemizeBetweenBrackets"] = mistv2_opts.phonemize_between_brackets

        headers = {
            "accept": f"audio/{format}",
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            format=format,
        )

        decode_task: asyncio.Task | None = None
        try:
            async with self._session.post(
                self._tts._base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    connect=self._conn_options.timeout,
                    total=self._total_timeout,
                ),
            ) as response:
                if not response.content_type.startswith("audio"):
                    content = await response.text()
                    logger.error("Rime returned non-audio data: %s", content)
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
