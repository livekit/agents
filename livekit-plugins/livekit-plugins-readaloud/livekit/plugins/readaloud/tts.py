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

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import quote

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSEngines, TTSSampleRates

DEFAULT_BASE_URL = "https://api.readaloudai.org"
DEFAULT_VOICE = "default"
DEFAULT_ENGINE: TTSEngines = "live"
DEFAULT_SAMPLE_RATE: TTSSampleRates = 24000
NUM_CHANNELS = 1

SAMPLE_RATES = (8000, 16000, 24000)
ENGINES = ("live", "studio")

# `POST /v1/text-to-speech` renders these natively. 16 kHz is only offered by the
# ElevenLabs-compatible route, `POST /v1/text-to-speech/{voice}/stream`.
_NATIVE_SAMPLE_RATES = (8000, 24000)

# Model ids the compatibility route uses to choose an engine for the default voice.
_COMPAT_MODEL_IDS: dict[str, str] = {
    "live": "eleven_flash_v2_5",
    "studio": "eleven_multilingual_v2",
}
_COMPAT_DEFAULT_VOICE = "readaloud-default"


@dataclass
class _TTSOptions:
    voice: str
    speed: float
    engine: TTSEngines
    sample_rate: int
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        engine: TTSEngines = DEFAULT_ENGINE,
        sample_rate: TTSSampleRates = DEFAULT_SAMPLE_RATE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of the ReadAloud TTS.

        Args:
            voice: Voice id. ``"default"`` is the English default voice; other ids
                (for example ``custom:<id>``) are listed by ``GET /v1/voices``.
            speed: Speaking rate multiplier. 1.0 is normal speed.
            engine: ``"live"`` (default, lowest latency and price, built for voice
                agents) or ``"studio"``.
            sample_rate: Output sample rate in Hz, one of 8000, 16000 or 24000. Use
                8000 for telephony.
            api_key: ReadAloud API key. Falls back to the ``READALOUD_API_KEY``
                environment variable.
            base_url: Override the ReadAloud API base URL.
            tokenizer: Sentence tokenizer used by ``stream()``. Defaults to
                ``tokenize.blingfire.SentenceTokenizer()``.
            http_session: Optional ``aiohttp.ClientSession`` to reuse.
        """
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")

        # ReadAloud has no bidirectional text socket and returns no word timings, so
        # `stream()` pipelines whole sentences over HTTP and transcripts are not aligned.
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        readaloud_api_key = api_key if is_given(api_key) else os.environ.get("READALOUD_API_KEY")
        if not readaloud_api_key:
            raise ValueError(
                "ReadAloud API key is required, either as `api_key` argument or "
                "`READALOUD_API_KEY` environment variable"
            )

        self._api_key = readaloud_api_key
        self._opts = _TTSOptions(
            voice=voice,
            speed=speed,
            engine=engine,
            sample_rate=sample_rate,
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return self._opts.engine

    @property
    def provider(self) -> str:
        return "ReadAloud"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


def _build_request(
    api_key: str, opts: _TTSOptions, text: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Return the URL, headers and JSON body for one synthesis request."""
    if opts.sample_rate in _NATIVE_SAMPLE_RATES:
        return (
            f"{opts.base_url}/v1/text-to-speech",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            {
                "text": text,
                "voice": opts.voice,
                "speed": opts.speed,
                "format": f"pcm_{opts.sample_rate}",
                "engine": opts.engine,
            },
        )

    voice = _COMPAT_DEFAULT_VOICE if opts.voice == DEFAULT_VOICE else opts.voice
    return (
        f"{opts.base_url}/v1/text-to-speech/{quote(voice, safe='')}/stream"
        f"?output_format=pcm_{opts.sample_rate}",
        {"xi-api-key": api_key, "Content-Type": "application/json"},
        {
            "text": text,
            "model_id": _COMPAT_MODEL_IDS[opts.engine],
            "voice_settings": {"speed": opts.speed},
        },
    )


async def _error_message(resp: aiohttp.ClientResponse) -> tuple[str, object | None]:
    """Read a native ``{"error": ...}`` or compat ``{"detail": ...}`` error body."""
    raw = await resp.text()
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:200] or f"HTTP {resp.status}", None

    message = ""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            message = str(detail.get("message") or "")
        elif isinstance(detail, list) and detail:
            first = detail[0]
            message = str(first.get("msg", first)) if isinstance(first, dict) else str(first)
        else:
            message = str(body.get("error") or body.get("message") or "")
    return message or f"HTTP {resp.status}", body


async def _synthesize(
    tts_: TTS,
    opts: _TTSOptions,
    text: str,
    output_emitter: tts.AudioEmitter,
    conn_options: APIConnectOptions,
) -> None:
    """Request one piece of text and push the audio to the emitter as it arrives.

    Leaving the ``async with`` before the body is fully read closes the connection,
    which is how a barge-in stops generation on the server.
    """
    url, headers, body = _build_request(tts_._api_key, opts, text)
    try:
        async with tts_._ensure_session().post(
            url,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(
                total=None, sock_connect=conn_options.timeout, sock_read=30
            ),
        ) as resp:
            request_id = resp.headers.get("x-request-id") or resp.headers.get("request-id")
            if resp.status != 200:
                message, error_body = await _error_message(resp)
                # 429 and 503 mean the API is at capacity; both are retryable.
                raise APIStatusError(
                    message=message,
                    status_code=resp.status,
                    request_id=request_id,
                    body=error_body,
                )

            async for data, _ in resp.content.iter_chunks():
                output_emitter.push(data)
    except asyncio.TimeoutError:
        raise APITimeoutError() from None
    except aiohttp.ClientResponseError as e:
        raise APIStatusError(
            message=e.message, status_code=e.status, request_id=None, body=None
        ) from None
    except aiohttp.ClientError as e:
        raise APIConnectionError() from e


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a complete string with a single streaming HTTP request."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        await _synthesize(
            self._tts, self._opts, self._input_text, output_emitter, self._conn_options
        )
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    """Split streamed LLM text into sentences and synthesize each one in order.

    The ReadAloud API takes a complete string per request, so streaming here is
    sentence-level: audio for the first sentence starts while the LLM is still
    producing the rest of the reply.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=segment_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=segment_id)

        sent_stream = self._tts._sentence_tokenizer.stream()

        async def _forward_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _synthesize_sentences() -> None:
            async for ev in sent_stream:
                text = ev.token.strip()
                if not text:
                    continue
                self._mark_started()
                await _synthesize(self._tts, self._opts, text, output_emitter, self._conn_options)
                output_emitter.flush()
            output_emitter.end_segment()

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize_sentences()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await sent_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)
