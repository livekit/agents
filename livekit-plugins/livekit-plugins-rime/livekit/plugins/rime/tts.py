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
import json
import os
import weakref
from dataclasses import dataclass, replace
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

from .langs import TTSLangs
from .log import logger
from .models import ArcanaVoices, DefaultCodaVoice, DefaultMistVoice, TTSModels

# arcana can take as long as 80% of the total duration of the audio it's synthesizing.
ARCANA_MODEL_TIMEOUT = 60 * 4
MIST_MODEL_TIMEOUT = 30
RIME_BASE_URL = "https://users.rime.ai/v1/rime-tts"
RIME_WS_BASE_URL = "wss://users-ws.rime.ai"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    arcana_options: _ArcanaOptions | None = None
    coda_options: _CodaOptions | None = None
    mist_options: _MistOptions | None = None


@dataclass
class _ArcanaOptions:
    repetition_penalty: NotGivenOr[float] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    top_p: NotGivenOr[float] = NOT_GIVEN
    max_tokens: NotGivenOr[int] = NOT_GIVEN
    lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    sample_rate: NotGivenOr[int] = NOT_GIVEN


@dataclass
class _CodaOptions:
    max_tokens: NotGivenOr[int] = NOT_GIVEN
    lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    sample_rate: NotGivenOr[int] = NOT_GIVEN


@dataclass
class _MistOptions:
    lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    sample_rate: NotGivenOr[int] = NOT_GIVEN
    speed_alpha: NotGivenOr[float] = NOT_GIVEN
    reduce_latency: NotGivenOr[bool] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN


def _is_mist_model(model: TTSModels | str) -> bool:
    return "mist" in model


def _timeout_for_model(model: TTSModels | str) -> int:
    if model == "arcana" or model == "coda":
        return ARCANA_MODEL_TIMEOUT
    return MIST_MODEL_TIMEOUT


def _model_params(opts: _TTSOptions) -> dict[str, object]:
    """Per-model option fields shared between the HTTP body and the WS query string."""
    params: dict[str, object] = {}
    if opts.model == "arcana" and opts.arcana_options is not None:
        ao = opts.arcana_options
        if is_given(ao.lang):
            params["lang"] = ao.lang
        if is_given(ao.repetition_penalty):
            params["repetition_penalty"] = ao.repetition_penalty
        if is_given(ao.temperature):
            params["temperature"] = ao.temperature
        if is_given(ao.top_p):
            params["top_p"] = ao.top_p
        if is_given(ao.max_tokens):
            params["max_tokens"] = ao.max_tokens
    elif opts.model == "coda" and opts.coda_options is not None:
        co = opts.coda_options
        if is_given(co.lang):
            params["lang"] = co.lang
        if is_given(co.max_tokens):
            params["max_tokens"] = co.max_tokens
    elif _is_mist_model(opts.model) and opts.mist_options is not None:
        mo = opts.mist_options
        if is_given(mo.lang):
            params["lang"] = mo.lang
        if is_given(mo.speed_alpha):
            params["speedAlpha"] = mo.speed_alpha
        if is_given(mo.pause_between_brackets):
            params["pauseBetweenBrackets"] = mo.pause_between_brackets
        if is_given(mo.phonemize_between_brackets):
            params["phonemizeBetweenBrackets"] = mo.phonemize_between_brackets
    return params


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "arcana",
        speaker: NotGivenOr[ArcanaVoices | str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        # Arcana options
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        # Mistv2 options
        sample_rate: int = 22050,
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        use_websocket: bool = False,
        segment: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        if is_given(base_url):
            # Infer streaming mode from URL prefix; an explicit use_websocket=True still wins.
            use_websocket = use_websocket or base_url.startswith(("ws://", "wss://"))
            resolved_base_url = base_url
        else:
            resolved_base_url = RIME_WS_BASE_URL if use_websocket else RIME_BASE_URL

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=use_websocket,
                aligned_transcript=use_websocket,
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
            if _is_mist_model(model):
                speaker = DefaultMistVoice
            elif model == "coda":
                speaker = DefaultCodaVoice
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
                lang=lang,
                sample_rate=sample_rate,
            )
        elif model == "coda":
            self._opts.coda_options = _CodaOptions(
                max_tokens=max_tokens,
                lang=lang,
                sample_rate=sample_rate,
            )
        elif _is_mist_model(model):
            self._opts.mist_options = _MistOptions(
                lang=lang,
                sample_rate=sample_rate,
                speed_alpha=speed_alpha,
                reduce_latency=reduce_latency,
                pause_between_brackets=pause_between_brackets,
                phonemize_between_brackets=phonemize_between_brackets,
            )
        self._session = http_session
        self._base_url = resolved_base_url
        self._use_websocket = use_websocket
        self._segment = segment if is_given(segment) else "bySentence"

        self._total_timeout = _timeout_for_model(model)

        self._streams: weakref.WeakSet[SynthesizeStream] = weakref.WeakSet()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
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
            "samplingRate": self._sample_rate,
            "segment": self._segment,
            **_model_params(self._opts),
        }
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
            logger.warning(f"Error during Rime WS close sequence: {e}")
        finally:
            await ws.close()

    def prewarm(self) -> None:
        if self._use_websocket:
            self._pool.prewarm()

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if not self._use_websocket:
            raise RuntimeError(
                "Rime TTS streaming requires use_websocket=True at construction time"
            )
        s = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(s)
        return s

    async def aclose(self) -> None:
        for s in list(self._streams):
            await s.aclose()
        self._streams.clear()
        await self._pool.aclose()

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
        # Arcana parameters
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        # Mistv2 parameters
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        # WS URL is bound at pool connect; invalidate if any URL-affecting param changed.
        prev_ws_url = self._ws_url() if self._use_websocket else None
        if is_given(base_url):
            self._base_url = base_url
        if is_given(model):
            self._opts.model = model
            self._total_timeout = _timeout_for_model(model)

            if model == "arcana" and self._opts.arcana_options is None:
                self._opts.arcana_options = _ArcanaOptions()
            elif model == "coda" and self._opts.coda_options is None:
                self._opts.coda_options = _CodaOptions()
            elif _is_mist_model(model) and self._opts.mist_options is None:
                self._opts.mist_options = _MistOptions()

        if is_given(speaker):
            self._opts.speaker = speaker

        if self._opts.model == "arcana" and self._opts.arcana_options is not None:
            if is_given(repetition_penalty):
                self._opts.arcana_options.repetition_penalty = repetition_penalty
            if is_given(temperature):
                self._opts.arcana_options.temperature = temperature
            if is_given(top_p):
                self._opts.arcana_options.top_p = top_p
            if is_given(max_tokens):
                self._opts.arcana_options.max_tokens = max_tokens
            if is_given(lang):
                self._opts.arcana_options.lang = lang
            if is_given(sample_rate):
                self._opts.arcana_options.sample_rate = sample_rate

        elif self._opts.model == "coda" and self._opts.coda_options is not None:
            if is_given(max_tokens):
                self._opts.coda_options.max_tokens = max_tokens
            if is_given(lang):
                self._opts.coda_options.lang = lang
            if is_given(sample_rate):
                self._opts.coda_options.sample_rate = sample_rate

        elif _is_mist_model(self._opts.model) and self._opts.mist_options is not None:
            if is_given(lang):
                self._opts.mist_options.lang = lang
            if is_given(sample_rate):
                self._opts.mist_options.sample_rate = sample_rate
            if is_given(speed_alpha):
                self._opts.mist_options.speed_alpha = speed_alpha
            if is_given(reduce_latency):
                self._opts.mist_options.reduce_latency = reduce_latency
            if is_given(pause_between_brackets):
                self._opts.mist_options.pause_between_brackets = pause_between_brackets
            if is_given(phonemize_between_brackets):
                self._opts.mist_options.phonemize_between_brackets = phonemize_between_brackets

        if prev_ws_url is not None and self._ws_url() != prev_ws_url:
            self._pool.invalidate()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(self, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.model,
            **_model_params(self._opts),
        }
        format = "audio/pcm"
        if self._opts.model == "arcana" and self._opts.arcana_options is not None:
            if is_given(self._opts.arcana_options.sample_rate):
                payload["samplingRate"] = self._opts.arcana_options.sample_rate
        elif self._opts.model == "coda" and self._opts.coda_options is not None:
            if is_given(self._opts.coda_options.sample_rate):
                payload["samplingRate"] = self._opts.coda_options.sample_rate
        elif _is_mist_model(self._opts.model) and self._opts.mist_options is not None:
            mist_opts = self._opts.mist_options
            if is_given(mist_opts.sample_rate):
                payload["samplingRate"] = mist_opts.sample_rate
            if self._opts.model == "mistv2" and is_given(mist_opts.reduce_latency):
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
                    logger.error("Rime returned non-audio data: %s", content)
                    return

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
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


class SynthesizeStream(tts.SynthesizeStream):
    """One stream = one utterance. Server-side bySentence segmentation by default;
    pass segment="immediate" on the TTS to disable server buffering when the agent
    is already feeding sentence-tokenized text."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
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

        sent_stream = self._tts._sentence_tokenizer.stream()
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
                    raise APIConnectionError(f"Rime ws error: {ws.exception()}")
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
                    msg_text = data.get("message", "(no message)")
                    raise APIError(f"Rime ws error: {msg_text}")

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
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
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(f"Rime WS error: {e}") from e
