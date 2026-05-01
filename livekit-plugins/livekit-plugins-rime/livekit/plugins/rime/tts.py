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
from .models import ArcanaVoices, DefaultMistVoice, TTSModels

# Arcana can take as long as 80% of the total audio duration to synthesize.
ARCANA_MODEL_TIMEOUT = 60 * 4
MIST_MODEL_TIMEOUT = 30

RIME_HTTP_BASE_URL = "https://users.rime.ai/v1/rime-tts"
RIME_WS_BASE_URL = "wss://users-ws.rime.ai/ws3"

NUM_CHANNELS = 1


def _is_mist_model(model: TTSModels | str) -> bool:
    return "mist" in model


def _timeout_for_model(model: TTSModels | str) -> int:
    return ARCANA_MODEL_TIMEOUT if model == "arcana" else MIST_MODEL_TIMEOUT


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    sample_rate: int
    lang: NotGivenOr[TTSLangs | str]
    word_timestamps: bool
    # WS buffer flush strategy: "immediate", "bySentence", or "never"
    segment: str
    # Mist-only params (HTTP + WS)
    speed_alpha: NotGivenOr[float]
    reduce_latency: NotGivenOr[bool]
    pause_between_brackets: NotGivenOr[bool]
    phonemize_between_brackets: NotGivenOr[bool]
    # Arcana-only sampling params (HTTP path only)
    repetition_penalty: NotGivenOr[float]
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    max_tokens: NotGivenOr[int]

    def ws_url(self, base_url: str) -> str:
        """Build the full WebSocket URL with connection parameters."""
        params: dict[str, str | int | float] = {
            "speaker": self.speaker,
            "modelId": self.model,
            "audioFormat": "pcm",
            "samplingRate": self.sample_rate,
            "segment": self.segment,
        }
        if is_given(self.lang):
            params["lang"] = self.lang  # type: ignore[assignment]
        if _is_mist_model(self.model):
            if is_given(self.speed_alpha):
                params["speedAlpha"] = self.speed_alpha  # type: ignore[assignment]
            if is_given(self.pause_between_brackets):
                params["pauseBetweenBrackets"] = str(self.pause_between_brackets).lower()
            if is_given(self.phonemize_between_brackets):
                params["phonemizeBetweenBrackets"] = str(
                    self.phonemize_between_brackets
                ).lower()
        return f"{base_url}?{urlencode(params)}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "arcana",
        speaker: NotGivenOr[ArcanaVoices | str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        sample_rate: int = 22050,
        word_timestamps: bool = True,
        # WS segmentation mode used by stream()
        segment: str = "immediate",
        # Mist-only params
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        # Arcana-only params — used by synthesize() (HTTP) only
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        http_base_url: str = RIME_HTTP_BASE_URL,
        ws_base_url: str = RIME_WS_BASE_URL,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of Rime TTS.

        Supports two synthesis paths:

        - ``synthesize()`` — single-shot HTTP request, returns a ``ChunkedStream``.
        - ``stream()`` — low-latency WebSocket streaming via ``/ws3``, returns a
          ``SynthesizeStream``.

        See https://docs.rime.ai/docs/websockets for WebSocket API details.

        Args:
            model: Rime model to use (``"arcana"``, ``"mistv2"``, ``"mistv3"``).
                Defaults to ``"arcana"``.
            speaker: Voice ID. Defaults to ``"astra"`` for Arcana and ``"cove"`` for Mist.
            lang: BCP-47-style language code (e.g. ``"eng"``, ``"spa"``). Defaults to ``"eng"``.
            sample_rate: Output audio sample rate in Hz. Defaults to ``22050``.
            word_timestamps: Request word-level timestamps (WebSocket path only).
                Defaults to ``True``.
            segment: WebSocket buffer flush strategy — ``"immediate"`` (flush after every
                text chunk), ``"bySentence"`` (flush after punctuation), or ``"never"``.
                Defaults to ``"immediate"``.
            speed_alpha: Speech tempo multiplier (Mist models only).
            reduce_latency: Reduce synthesis latency at the cost of quality
                (``mistv2`` HTTP path only).
            pause_between_brackets: Insert pauses at bracket-delimited segments.
            phonemize_between_brackets: Apply custom pronunciation inside brackets.
            repetition_penalty: Sampling repetition penalty (Arcana, HTTP path only).
            temperature: Sampling temperature (Arcana, HTTP path only).
            top_p: Nucleus sampling probability (Arcana, HTTP path only).
            max_tokens: Maximum tokens to generate (Arcana, HTTP path only).
            api_key: Rime API key. Falls back to the ``RIME_API_KEY`` environment variable.
            http_session: Optional existing ``aiohttp.ClientSession``.
            http_base_url: Override for the HTTP endpoint URL.
            ws_base_url: Override for the WebSocket endpoint URL.
            tokenizer: Sentence tokenizer used for WebSocket streaming. Defaults to
                ``tokenize.blingfire.SentenceTokenizer``.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key if is_given(api_key) else os.environ.get("RIME_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Rime API key is required, either as argument or set RIME_API_KEY "
                "environment variable"
            )

        if not is_given(speaker):
            speaker = DefaultMistVoice if _is_mist_model(model) else "astra"

        self._opts = _TTSOptions(
            model=model,
            speaker=speaker,
            sample_rate=sample_rate,
            lang=lang,
            word_timestamps=word_timestamps,
            segment=segment,
            speed_alpha=speed_alpha,
            reduce_latency=reduce_latency,
            pause_between_brackets=pause_between_brackets,
            phonemize_between_brackets=phonemize_between_brackets,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self._http_base_url = http_base_url
        self._ws_base_url = ws_base_url
        self._session = http_session
        self._total_timeout = _timeout_for_model(model)

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
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

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        url = self._opts.ws_url(self._ws_base_url)
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(
                url,
                headers={"Authorization": f"Bearer {self._api_key}"},
            ),
            timeout,
        )
        logger.debug("Established Rime WebSocket connection", extra={"url": url})
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
        segment: NotGivenOr[str] = NOT_GIVEN,
        # Mist params
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        # Arcana HTTP-only params
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        ws_params_changed = False

        if is_given(model):
            self._opts.model = model
            self._total_timeout = _timeout_for_model(model)
            ws_params_changed = True
        if is_given(speaker):
            self._opts.speaker = speaker
            ws_params_changed = True
        if is_given(lang):
            self._opts.lang = lang
            ws_params_changed = True
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            ws_params_changed = True
        if is_given(word_timestamps):
            self._opts.word_timestamps = word_timestamps
        if is_given(segment):
            self._opts.segment = segment
            ws_params_changed = True
        if is_given(speed_alpha):
            self._opts.speed_alpha = speed_alpha
            ws_params_changed = True
        if is_given(reduce_latency):
            self._opts.reduce_latency = reduce_latency
        if is_given(pause_between_brackets):
            self._opts.pause_between_brackets = pause_between_brackets
            ws_params_changed = True
        if is_given(phonemize_between_brackets):
            self._opts.phonemize_between_brackets = phonemize_between_brackets
            ws_params_changed = True
        if is_given(repetition_penalty):
            self._opts.repetition_penalty = repetition_penalty
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(top_p):
            self._opts.top_p = top_p
        if is_given(max_tokens):
            self._opts.max_tokens = max_tokens

        if ws_params_changed:
            # Recycle the pool so the next stream reconnects with the updated URL.
            old_pool = self._pool
            self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
                connect_cb=self._connect_ws,
                close_cb=self._close_ws,
                max_session_duration=300,
                mark_refreshed_on_get=True,
            )
            asyncio.ensure_future(old_pool.aclose())

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SynthesizeStream":
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Single-shot HTTP synthesis via the Rime REST endpoint."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.model,
        }
        mime_type = "audio/pcm"

        if self._opts.model == "arcana":
            if is_given(self._opts.repetition_penalty):
                payload["repetition_penalty"] = self._opts.repetition_penalty
            if is_given(self._opts.temperature):
                payload["temperature"] = self._opts.temperature
            if is_given(self._opts.top_p):
                payload["top_p"] = self._opts.top_p
            if is_given(self._opts.max_tokens):
                payload["max_tokens"] = self._opts.max_tokens
            if is_given(self._opts.lang):
                payload["lang"] = self._opts.lang
            payload["samplingRate"] = self._opts.sample_rate
        elif _is_mist_model(self._opts.model):
            if is_given(self._opts.lang):
                payload["lang"] = self._opts.lang
            payload["samplingRate"] = self._opts.sample_rate
            if is_given(self._opts.speed_alpha):
                payload["speedAlpha"] = self._opts.speed_alpha
            if self._opts.model == "mistv2" and is_given(self._opts.reduce_latency):
                payload["reduceLatency"] = self._opts.reduce_latency
            if is_given(self._opts.pause_between_brackets):
                payload["pauseBetweenBrackets"] = self._opts.pause_between_brackets
            if is_given(self._opts.phonemize_between_brackets):
                payload["phonemizeBetweenBrackets"] = self._opts.phonemize_between_brackets

        try:
            async with self._tts._ensure_session().post(
                self._tts._http_base_url,
                headers={
                    "accept": mime_type,
                    "Authorization": f"Bearer {self._tts._api_key}",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self._tts._total_timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio"):
                    body = await resp.text()
                    logger.error("Rime returned non-audio response: %s", body)
                    raise APIConnectionError(f"unexpected content-type: {resp.content_type}")

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=mime_type,
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Low-latency WebSocket streaming synthesis via the Rime ``/ws3`` endpoint."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        context_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        if self._opts.model == "arcana" and any(
            is_given(p)
            for p in (
                self._opts.repetition_penalty,
                self._opts.temperature,
                self._opts.top_p,
                self._opts.max_tokens,
            )
        ):
            logger.warning(
                "Arcana sampling params (repetition_penalty, temperature, top_p, max_tokens) "
                "are not supported on the WebSocket path and will be ignored. "
                "Use synthesize() for the HTTP path if you need these controls."
            )

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        input_sent_event = asyncio.Event()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for ev in sent_tokenizer_stream:
                pkt = {"text": ev.token + " ", "contextId": context_id}
                self._mark_started()
                await ws.send_str(json.dumps(pkt))
                input_sent_event.set()

            # Flush any content remaining in the Rime buffer.
            await ws.send_str(json.dumps({"operation": "flush"}))
            input_sent_event.set()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Wait until at least one packet has been sent before polling.
            await input_sent_event.wait()

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.error(
                        "Rime WebSocket connection closed unexpectedly",
                        extra={"context_id": context_id},
                    )
                    raise APIStatusError(
                        "Rime WebSocket connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Rime message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "chunk":
                    raw = base64.b64decode(data["data"])
                    output_emitter.push(raw)

                elif msg_type == "timestamps":
                    word_ts = data.get("word_timestamps", {})
                    words = word_ts.get("words", [])
                    starts = word_ts.get("start", [])
                    ends = word_ts.get("end", [])
                    for word, start, end in zip(words, starts, ends, strict=False):
                        output_emitter.push_timed_transcript(
                            TimedString(text=word, start_time=start, end_time=end)
                        )

                elif msg_type == "done":
                    output_emitter.end_input()
                    break

                elif msg_type == "error":
                    raise APIError(f"Rime error: {data.get('message', data)}")

                else:
                    logger.debug("unhandled Rime message: %s", data)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused

                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except (APITimeoutError, APIStatusError, APIError, APIConnectionError):
            raise
        except Exception as e:
            logger.exception("Rime WebSocket error", extra={"context_id": context_id})
            raise APIConnectionError() from e
