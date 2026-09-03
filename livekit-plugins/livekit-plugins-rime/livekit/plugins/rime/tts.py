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
import base64
import copy
import json
import os
import weakref
from dataclasses import dataclass
from typing import Literal, overload
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from ._websocket_v1 import (
    DEFAULT_AUDIO_FORMAT,
    RimeAudioFormat,
    WebSocketProtocol,
    model_from_websocket_url,
    validate_audio_format,
    validate_endpoint_host,
)
from ._websocket_v1_adapter import V1SynthesisOptions, WebSocketV1Adapter
from .langs import TTSLangs
from .log import logger
from .models import (
    MODEL_CODA,
    MODEL_MIST_V3,
    DefaultCodaVoice,
    DefaultMistVoice,
    TTSModels,
    is_mist_model,
    supports_reduce_latency,
    supports_time_scale_factor,
)

CODA_MODEL_TIMEOUT = 60 * 4
MIST_MODEL_TIMEOUT = 30
RIME_BASE_URL = "https://users.rime.ai/v1/rime-tts"
RIME_WS_BASE_URL = "wss://users-ws.rime.ai"
NUM_CHANNELS = 1
CODA_DEFAULT_SAMPLE_RATE = 24000
MIST_V2_DEFAULT_SAMPLE_RATE = 22050
MIST_V3_DEFAULT_SAMPLE_RATE = 24000


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    language: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    audio_format: RimeAudioFormat = DEFAULT_AUDIO_FORMAT
    sample_rate: NotGivenOr[int] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN
    coda_options: _CodaOptions | None = None
    mist_options: _MistOptions | None = None


@dataclass
class _CodaOptions:
    repetition_penalty: NotGivenOr[float] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    top_p: NotGivenOr[float] = NOT_GIVEN
    max_tokens: NotGivenOr[int] = NOT_GIVEN
    speed_alpha: NotGivenOr[float] = NOT_GIVEN


@dataclass
class _MistOptions:
    speed_alpha: NotGivenOr[float] = NOT_GIVEN
    reduce_latency: NotGivenOr[bool] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN


def _timeout_for_model(model: TTSModels | str) -> int:
    if model == MODEL_CODA:
        return CODA_MODEL_TIMEOUT
    return MIST_MODEL_TIMEOUT


def _default_sample_rate(model: TTSModels | str) -> int:
    if model == MODEL_CODA:
        return CODA_DEFAULT_SAMPLE_RATE
    if model == MODEL_MIST_V3:
        return MIST_V3_DEFAULT_SAMPLE_RATE
    return MIST_V2_DEFAULT_SAMPLE_RATE


def _requested_sample_rate(options: _TTSOptions) -> NotGivenOr[int]:
    return options.sample_rate


def _model_params(opts: _TTSOptions) -> dict[str, object]:
    """Per-model option fields shared between the HTTP body and the WS query string."""
    params: dict[str, object] = {}
    if is_given(opts.language):
        params["lang"] = opts.language
    if is_given(opts.time_scale_factor) and supports_time_scale_factor(opts.model):
        params["timeScaleFactor"] = opts.time_scale_factor
    if opts.model == MODEL_CODA and opts.coda_options is not None:
        co = opts.coda_options
        if is_given(co.repetition_penalty):
            params["repetition_penalty"] = co.repetition_penalty
        if is_given(co.temperature):
            params["temperature"] = co.temperature
        if is_given(co.top_p):
            params["top_p"] = co.top_p
        if is_given(co.max_tokens):
            params["max_tokens"] = co.max_tokens
        if is_given(co.speed_alpha):
            params["speedAlpha"] = co.speed_alpha
    elif is_mist_model(opts.model) and opts.mist_options is not None:
        mo = opts.mist_options
        if is_given(mo.speed_alpha):
            params["speedAlpha"] = mo.speed_alpha
        if is_given(mo.pause_between_brackets):
            params["pauseBetweenBrackets"] = mo.pause_between_brackets
        if is_given(mo.phonemize_between_brackets):
            params["phonemizeBetweenBrackets"] = mo.phonemize_between_brackets
    return params


def _check_time_scale_factor_supported(
    model: TTSModels | str, time_scale_factor: NotGivenOr[float]
) -> None:
    if is_given(time_scale_factor) and not supports_time_scale_factor(model):
        raise ValueError("time_scale_factor is not supported by the mistv2 model")


def _resolve_websocket_model(
    websocket_url: str,
    model: NotGivenOr[TTSModels | str],
    *,
    allow_custom_endpoint: bool,
    current_model: TTSModels | str | None = None,
) -> TTSModels | str:
    endpoint_model = model_from_websocket_url(
        websocket_url, allow_custom_endpoint=allow_custom_endpoint
    )
    if endpoint_model is None:
        if not is_given(model):
            if current_model is not None:
                return current_model
            raise ValueError("model is required when websocket_url ends with /ws")
        return model
    if is_given(model):
        raise ValueError("model is derived from websocket_url; omit model")
    if endpoint_model == "mist":
        return MODEL_MIST_V3
    return endpoint_model


class TTS(tts.TTS[Literal["rime_tts_event"]]):
    @overload
    def __init__(
        self,
        *,
        websocket_url: str,
        websocket_protocol: WebSocketProtocol = "binary",
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        audio_format: RimeAudioFormat = DEFAULT_AUDIO_FORMAT,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        allow_custom_endpoint: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        use_websocket: bool = False,
        segment: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        allow_custom_endpoint: bool = False,
    ) -> None: ...

    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        websocket_url: NotGivenOr[str] = NOT_GIVEN,
        websocket_protocol: WebSocketProtocol = "binary",
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        audio_format: NotGivenOr[RimeAudioFormat | str] = NOT_GIVEN,
        # Coda options
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        # Shared by Mist and Coda (HTTP and v1 WebSocket)
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        # Supported by HTTP and the legacy ws3 interface
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        # Supported by all models
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        use_websocket: bool = False,
        segment: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        allow_custom_endpoint: bool = False,
    ) -> None:
        websocket_v1_url = websocket_url if is_given(websocket_url) else None
        if websocket_v1_url is None and is_given(audio_format):
            raise ValueError("audio_format is only supported with the Rime v1 WebSocket interface")
        resolved_audio_format = (
            validate_audio_format(audio_format) if is_given(audio_format) else DEFAULT_AUDIO_FORMAT
        )
        if websocket_v1_url is not None:
            if is_given(base_url):
                raise ValueError("websocket_url cannot be used with base_url")
            if use_websocket:
                raise ValueError("websocket_url enables WebSocket streaming; omit use_websocket")
            if is_given(speed_alpha):
                raise ValueError(
                    "speed_alpha belongs to the legacy Rime interfaces; use time_scale_factor"
                )
            if any(
                is_given(value) for value in (repetition_penalty, temperature, top_p, max_tokens)
            ):
                raise ValueError(
                    "generation controls are not supported by the Rime v1 WebSocket protocol"
                )
            if is_given(reduce_latency) or is_given(segment):
                raise ValueError("websocket_url cannot be used with ws3-only options")
            use_websocket = True
            resolved_base_url = RIME_BASE_URL
        elif is_given(base_url):
            validate_endpoint_host(base_url, allow_custom_endpoint=allow_custom_endpoint)
            # Infer streaming mode from URL prefix; an explicit use_websocket=True still wins.
            use_websocket = use_websocket or base_url.startswith(("ws://", "wss://"))
            resolved_base_url = base_url
        else:
            resolved_base_url = RIME_WS_BASE_URL if use_websocket else RIME_BASE_URL

        if websocket_v1_url is not None:
            resolved_model = _resolve_websocket_model(
                websocket_v1_url,
                model,
                allow_custom_endpoint=allow_custom_endpoint,
            )
            model_is_explicit = resolved_model != MODEL_CODA
        elif is_given(model):
            resolved_model = model
            model_is_explicit = True
        else:
            resolved_model = MODEL_CODA
            model_is_explicit = False

        _check_time_scale_factor_supported(resolved_model, time_scale_factor)
        if (
            websocket_v1_url is not None
            and not is_mist_model(resolved_model)
            and any(
                is_given(value) for value in (pause_between_brackets, phonemize_between_brackets)
            )
        ):
            raise ValueError("Mist options require a Mist model")
        resolved_sample_rate = (
            sample_rate if is_given(sample_rate) else _default_sample_rate(resolved_model)
        )
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=use_websocket,
                aligned_transcript=use_websocket and websocket_v1_url is None,
            ),
            sample_rate=resolved_sample_rate,
            num_channels=NUM_CHANNELS,
        )
        resolved_api_key = api_key if is_given(api_key) else os.environ.get("RIME_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Rime API key is required, either as argument or set RIME_API_KEY environmental variable"  # noqa: E501
            )
        self._api_key = resolved_api_key
        self._allow_custom_endpoint = allow_custom_endpoint

        if not is_given(speaker):
            if not model_is_explicit:
                speaker = "astra"
            elif is_mist_model(resolved_model):
                speaker = DefaultMistVoice
            elif resolved_model == MODEL_CODA:
                speaker = DefaultCodaVoice
            else:
                speaker = "astra"

        self._opts = _TTSOptions(
            model=resolved_model,
            speaker=speaker,
            language=lang,
            audio_format=resolved_audio_format,
            sample_rate=sample_rate,
            time_scale_factor=time_scale_factor,
        )
        if resolved_model == MODEL_CODA:
            self._opts.coda_options = _CodaOptions(
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                speed_alpha=speed_alpha,
            )
        elif is_mist_model(resolved_model):
            self._opts.mist_options = _MistOptions(
                speed_alpha=speed_alpha,
                reduce_latency=reduce_latency,
                pause_between_brackets=pause_between_brackets,
                phonemize_between_brackets=phonemize_between_brackets,
            )
        self._session = http_session
        self._base_url = resolved_base_url
        self._use_websocket = use_websocket
        self._segment = segment if is_given(segment) else "bySentence"
        self._sentence_tokenizer: tokenize.SentenceTokenizer | None = None
        if websocket_v1_url is None:
            self._sentence_tokenizer = (
                tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
            )
        self._websocket_v1_adapter = (
            WebSocketV1Adapter(
                websocket_v1_url=websocket_v1_url,
                websocket_protocol=websocket_protocol,
                api_key=self._api_key,
                ensure_session=self._ensure_session,
                sentence_tokenizer=tokenizer if is_given(tokenizer) else None,
                allow_custom_endpoint=allow_custom_endpoint,
            )
            if websocket_v1_url is not None
            else None
        )

        self._total_timeout = _timeout_for_model(resolved_model)

        self._streams: weakref.WeakSet[tts.SynthesizeStream] = weakref.WeakSet()
        self._ws3_pool = (
            self._new_ws3_pool()
            if self._use_websocket and self._websocket_v1_adapter is None
            else None
        )

    def _new_ws3_pool(self) -> utils.ConnectionPool[aiohttp.ClientWebSocketResponse]:
        return utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Rime"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def _ws_url(self) -> str:
        params: dict[str, object] = {
            "speaker": self._opts.speaker,
            "modelId": self._opts.model,
            "audioFormat": "pcm",
            "segment": self._segment,
            **_model_params(self._opts),
        }
        requested_sample_rate = self._opts.sample_rate
        if is_given(requested_sample_rate):
            params["samplingRate"] = requested_sample_rate
        encoded = {
            k: ("true" if v else "false") if isinstance(v, bool) else v for k, v in params.items()
        }
        return f"{self._base_url}/ws3?{urlencode(encoded)}"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(
            session.ws_connect(
                self._ws_url(), headers={"Authorization": f"Bearer {self._api_key}"}
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await ws.send_str(json.dumps({"operation": "eos"}))
            try:
                await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            logger.warning(
                "error during Rime WebSocket close sequence",
                extra={"exception_type": type(e).__name__},
            )
        finally:
            await ws.close()

    def prewarm(self) -> None:
        if self._websocket_v1_adapter is not None:
            self._websocket_v1_adapter.prewarm()
        elif self._use_websocket:
            assert self._ws3_pool is not None
            self._ws3_pool.prewarm()

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        if not self._use_websocket:
            raise RuntimeError(
                "Rime TTS streaming requires use_websocket=True at construction time"
            )
        s: tts.SynthesizeStream
        if self._websocket_v1_adapter is not None:
            s = self._websocket_v1_adapter.stream(
                tts_instance=self,
                options=self._v1_synthesis_options(),
                conn_options=conn_options,
            )
        else:
            s = _WS3SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(s)
        return s

    def _v1_synthesis_options(self) -> V1SynthesisOptions:
        mist = self._opts.mist_options if is_mist_model(self._opts.model) else None
        return V1SynthesisOptions(
            model=self._opts.model,
            speaker=self._opts.speaker,
            language=(str(self._opts.language) if is_given(self._opts.language) else NOT_GIVEN),
            audio_format=self._opts.audio_format,
            sampling_rate=self.sample_rate,
            time_scale_factor=(
                self._opts.time_scale_factor
                if supports_time_scale_factor(self._opts.model)
                else NOT_GIVEN
            ),
            pause_between_brackets=(mist.pause_between_brackets if mist is not None else NOT_GIVEN),
            phonemize_between_brackets=(
                mist.phonemize_between_brackets if mist is not None else NOT_GIVEN
            ),
        )

    async def aclose(self) -> None:
        for s in list(self._streams):
            await s.aclose()
        self._streams.clear()
        if self._websocket_v1_adapter is not None:
            await self._websocket_v1_adapter.aclose()
        elif self._ws3_pool is not None:
            await self._ws3_pool.aclose()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        if self._use_websocket:
            raise RuntimeError(
                "Rime TTS one-shot synthesize requires use_websocket=False at construction time"
            )
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN,
        # Coda parameters
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        audio_format: NotGivenOr[RimeAudioFormat | str] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        # Mistv2 parameters
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        websocket_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        updated_audio_format: RimeAudioFormat | None = None
        if self._websocket_v1_adapter is not None:
            if is_given(audio_format):
                updated_audio_format = validate_audio_format(audio_format)
            if is_given(model) and not is_given(websocket_url):
                raise ValueError(
                    "model can only be updated together with websocket_url for Rime v1"
                )
            if is_given(base_url):
                raise ValueError("use websocket_url to update a Rime v1 endpoint")
            if is_given(speed_alpha) or is_given(reduce_latency):
                raise ValueError("Rime v1 cannot be updated with ws3-only options")
            if any(
                is_given(value) for value in (repetition_penalty, temperature, top_p, max_tokens)
            ):
                raise ValueError("Rime v1 does not support generation controls")
            effective_model = self._opts.model
            if is_given(websocket_url):
                effective_model = _resolve_websocket_model(
                    websocket_url,
                    model,
                    allow_custom_endpoint=self._allow_custom_endpoint,
                    current_model=self._opts.model,
                )
            if not is_mist_model(effective_model) and any(
                is_given(value) for value in (pause_between_brackets, phonemize_between_brackets)
            ):
                raise ValueError("Mist options require a Mist model")
        elif is_given(websocket_url):
            raise ValueError("websocket_url can only update a TTS constructed with websocket_url")
        else:
            if is_given(audio_format):
                raise ValueError(
                    "audio_format is only supported with the Rime v1 WebSocket interface"
                )
            effective_model = model if is_given(model) else self._opts.model

        if is_given(base_url):
            validate_endpoint_host(base_url, allow_custom_endpoint=self._allow_custom_endpoint)

        _check_time_scale_factor_supported(effective_model, time_scale_factor)

        # The WS URL is bound when its pool connects. Refresh the pool when that URL changes.
        prev_ws_url = (
            self._ws_url() if self._use_websocket and self._websocket_v1_adapter is None else None
        )
        if is_given(websocket_url):
            assert self._websocket_v1_adapter is not None
            self._websocket_v1_adapter.update_endpoint(websocket_url)
            self._opts.model = effective_model
            self._total_timeout = _timeout_for_model(effective_model)
            if effective_model == MODEL_CODA and self._opts.coda_options is None:
                self._opts.coda_options = _CodaOptions()
            elif is_mist_model(effective_model) and self._opts.mist_options is None:
                self._opts.mist_options = _MistOptions()
        if is_given(base_url):
            self._base_url = base_url
        if is_given(model):
            self._opts.model = model
            self._total_timeout = _timeout_for_model(model)

            if model == MODEL_CODA and self._opts.coda_options is None:
                self._opts.coda_options = _CodaOptions()
            elif is_mist_model(model) and self._opts.mist_options is None:
                self._opts.mist_options = _MistOptions()

        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(lang):
            self._opts.language = lang
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if updated_audio_format is not None:
            self._opts.audio_format = updated_audio_format
        if is_given(time_scale_factor):
            self._opts.time_scale_factor = time_scale_factor
        if self._opts.model == MODEL_CODA and self._opts.coda_options is not None:
            if is_given(repetition_penalty):
                self._opts.coda_options.repetition_penalty = repetition_penalty
            if is_given(temperature):
                self._opts.coda_options.temperature = temperature
            if is_given(top_p):
                self._opts.coda_options.top_p = top_p
            if is_given(max_tokens):
                self._opts.coda_options.max_tokens = max_tokens
            if is_given(speed_alpha):
                self._opts.coda_options.speed_alpha = speed_alpha

        elif is_mist_model(self._opts.model) and self._opts.mist_options is not None:
            if is_given(speed_alpha):
                self._opts.mist_options.speed_alpha = speed_alpha
            if is_given(reduce_latency):
                self._opts.mist_options.reduce_latency = reduce_latency
            if is_given(pause_between_brackets):
                self._opts.mist_options.pause_between_brackets = pause_between_brackets
            if is_given(phonemize_between_brackets):
                self._opts.mist_options.phonemize_between_brackets = phonemize_between_brackets

        requested_sample_rate = self._opts.sample_rate
        self._sample_rate = (
            requested_sample_rate
            if is_given(requested_sample_rate)
            else _default_sample_rate(self._opts.model)
        )

        if prev_ws_url is not None and self._ws_url() != prev_ws_url:
            assert self._ws3_pool is not None
            self._ws3_pool.invalidate()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(self, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        self._sample_rate = tts.sample_rate
        self._opts = copy.deepcopy(tts._opts)
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict[str, object] = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.model,
            **_model_params(self._opts),
        }
        format = "audio/pcm"
        requested_sample_rate = _requested_sample_rate(self._opts)
        if is_given(requested_sample_rate):
            payload["samplingRate"] = requested_sample_rate
        if is_mist_model(self._opts.model) and self._opts.mist_options is not None:
            mist_opts = self._opts.mist_options
            if supports_reduce_latency(self._opts.model) and is_given(mist_opts.reduce_latency):
                payload["reduceLatency"] = mist_opts.reduce_latency

        try:
            async with self._tts._ensure_session().post(
                self._tts._base_url,
                headers={
                    "accept": format,
                    "Authorization": f"Bearer {self._tts._api_key}",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self._tts._total_timeout, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio"):
                    content = await resp.text()
                    logger.error("Rime returned non-audio data", extra={"lk.pii.data": content})
                    return

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=format,
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class _WS3SynthesizeStream(tts.SynthesizeStream):
    """One stream = one utterance. Server-side bySentence segmentation by default;
    pass segment="immediate" on the TTS to disable server buffering when the agent
    is already feeding sentence-tokenized text."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = copy.deepcopy(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        await self._run_ws3(output_emitter)

    async def _run_ws3(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)

        sentence_tokenizer = self._tts._sentence_tokenizer
        assert sentence_tokenizer is not None
        sent_stream = sentence_tokenizer.stream()
        input_sent_event = asyncio.Event()
        empty_input = False

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal empty_input
            sent_count = 0
            async for ev in sent_stream:
                pkt = {"text": ev.token + " ", "contextId": context_id}
                self._mark_started()
                await ws.send_str(json.dumps(pkt))
                input_sent_event.set()
                sent_count += 1
            if sent_count == 0:
                empty_input = True
                input_sent_event.set()
                output_emitter.end_input()
                return
            await ws.send_str(json.dumps({"operation": "flush", "contextId": context_id}))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            if empty_input:
                return
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Rime ws closed unexpectedly",
                        request_id=request_id,
                    )
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("Rime WebSocket transport error")
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Rime ws message type %s", msg.type)
                    continue
                data = json.loads(msg.data)
                t = data.get("type")
                if t == "chunk":
                    output_emitter.push(base64.b64decode(data["data"]))
                elif t == "timestamps":
                    wt = data.get("word_timestamps") or {}
                    words = wt.get("words") or []
                    starts = wt.get("start") or []
                    ends = wt.get("end") or []
                    for w, s, e in zip(words, starts, ends, strict=False):
                        output_emitter.push_timed_transcript(
                            TimedString(text=w + " ", start_time=s, end_time=e)
                        )
                elif t == "done":
                    output_emitter.end_input()
                    break
                elif t == "error":
                    raise APIError("Rime WebSocket request failed")

        try:
            assert self._tts._ws3_pool is not None
            async with self._tts._ws3_pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await sent_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message="Rime WebSocket request failed",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except APIError:
            raise
        except Exception:
            raise APIConnectionError("Rime WebSocket request failed") from None
