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
from typing import Any

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

DEFAULT_BASE_URL = "https://api.bland.ai/v2"
DEFAULT_VOICE_ID = "f04af0e5-1a80-48a9-b02d-52f30d417cfa"
DEFAULT_SAMPLE_RATE = 48000
NUM_CHANNELS = 1

SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)


@dataclass
class _TTSOptions:
    voice_id: str
    sample_rate: int
    expressiveness: NotGivenOr[float]
    stability: NotGivenOr[float]
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of the Bland TTS.

        Args:
            voice_id: Bland voice UUID; names are not accepted. Defaults to a ``BTTS_V3``
                voice. ``BTTS_V2`` voices work, but the controls below are calibrated for
                ``BTTS_V3`` and newer.
            sample_rate: Output sample rate in Hz, one of 8000, 16000, 24000, 44100, 48000.
                Defaults to 48000, the rate ``BTTS_V3`` renders natively.
            expressiveness: 0.0-1.0. Higher is more varied intonation.
            stability: 0.0-1.0. Higher is more consistent between renders.
            api_key: Bland API key. Falls back to the ``BLAND_API_KEY`` environment variable.
            base_url: Override the Bland API base URL.
            http_session: Optional ``aiohttp.ClientSession`` to reuse.
        """
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        bland_api_key = api_key if is_given(api_key) else os.environ.get("BLAND_API_KEY")
        if not bland_api_key:
            raise ValueError(
                "Bland API key is required, either as `api_key` argument or "
                "`BLAND_API_KEY` environment variable"
            )

        self._api_key = bland_api_key
        self._opts = _TTSOptions(
            voice_id=voice_id,
            sample_rate=sample_rate,
            expressiveness=expressiveness,
            stability=stability,
            base_url=base_url,
        )
        self._session = http_session

    @property
    def provider(self) -> str:
        return "Bland"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(expressiveness):
            self._opts.expressiveness = expressiveness
        if is_given(stability):
            self._opts.stability = stability

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        controls: dict[str, float] = {}
        if is_given(self._opts.expressiveness):
            controls["expressiveness"] = self._opts.expressiveness
        if is_given(self._opts.stability):
            controls["stability"] = self._opts.stability

        body: dict[str, Any] = {
            "text": self._input_text,
            "voice": self._opts.voice_id,
            "audio": {"encoding": "pcm_s16le", "sample_rate": self._opts.sample_rate},
        }
        if controls:
            body["controls"] = controls

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/tts",
                headers={
                    "authorization": self._tts._api_key,
                    "content-type": "application/json",
                },
                json=body,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    raise APIStatusError(
                        message=await _error_message(resp),
                        status_code=resp.status,
                        request_id=resp.headers.get("x-request-id"),
                        body=None,
                    )

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id") or utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


async def _error_message(resp: aiohttp.ClientResponse) -> str:
    """Unwrap the v2 ``{"error": {"code", "message"}}`` envelope, falling back to the raw body."""
    try:
        payload = await resp.json()
    except Exception:
        return resp.reason or "request failed"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code, message = error.get("code"), error.get("message")
        if code and message:
            return f"{code}: {message}"
        if message:
            return str(message)
    return str(payload)
