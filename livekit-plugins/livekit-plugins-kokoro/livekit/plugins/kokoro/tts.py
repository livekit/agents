# Copyright 2026 LiveKit, Inc.
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
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSModels

DEFAULT_BASE_URL = "http://localhost:8880/v1"
DEFAULT_MODEL = "kokoro"
DEFAULT_VOICE = "af_heart"

# Kokoro-FastAPI always generates 24 kHz mono s16le audio; the server-side
# sample rate is not configurable.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    speed: float
    lang_code: NotGivenOr[str]
    base_url: str
    api_key: NotGivenOr[str]


async def list_voices(
    *,
    base_url: NotGivenOr[str] = NOT_GIVEN,
    api_key: NotGivenOr[str] = NOT_GIVEN,
    http_session: aiohttp.ClientSession | None = None,
) -> list[str]:
    """List the voices available on a Kokoro-FastAPI server.

    Args:
        base_url: Base URL of the Kokoro-FastAPI server, including the ``/v1``
            prefix. If not provided, uses the ``KOKORO_BASE_URL`` env variable,
            falling back to ``http://localhost:8880/v1``.
        api_key: Optional bearer token, for deployments behind an
            authenticating proxy. Kokoro-FastAPI itself requires none.
        http_session: Optional aiohttp session to use for the request.
    """
    resolved_base_url = _resolve_base_url(base_url)
    session = http_session or utils.http_context.http_session()
    async with session.get(
        f"{resolved_base_url}/audio/voices", headers=_auth_headers(api_key)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        voices = data["voices"] if isinstance(data, dict) else data
        # newer servers return [{"id": ..., "name": ...}], older ones ["af_heart", ...]
        return [v["id"] if isinstance(v, dict) else v for v in voices]


def _resolve_base_url(base_url: NotGivenOr[str]) -> str:
    if is_given(base_url):
        return base_url.rstrip("/")
    return os.environ.get("KOKORO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _auth_headers(api_key: NotGivenOr[str]) -> dict[str, str]:
    if is_given(api_key):
        return {"Authorization": f"Bearer {api_key}"}
    return {}


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model: TTSModels | str = DEFAULT_MODEL,
        speed: float = 1.0,
        lang_code: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Kokoro TTS, backed by a Kokoro-FastAPI server.

        Kokoro is an open-weight TTS model; `Kokoro-FastAPI
        <https://github.com/remsky/Kokoro-FastAPI>`_ serves it over an
        OpenAI-compatible HTTP API. Run one locally with::

            docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest  # CPU
            docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest  # NVIDIA

        Audio is requested as a raw PCM stream (24 kHz mono), so chunks are
        forwarded to the pipeline as they are generated.

        Args:
            voice: Voice ID, e.g. ``af_heart``. Kokoro voice blending syntax is
                supported, e.g. ``"af_bella(2)+af_sky(1)"``. Use the module-level
                ``list_voices()`` helper to discover available voices.
            model: Model name as exposed by the server; ``kokoro`` by default.
            speed: Speech speed multiplier, between 0.25 and 4.0.
            lang_code: Language code for phonemization (e.g. ``a`` for US
                English, ``b`` for British English, ``z`` for Mandarin). If not
                provided, the server infers it from the first letter of the
                voice name.
            base_url: Base URL of the Kokoro-FastAPI server, including the
                ``/v1`` prefix. If not provided, uses the ``KOKORO_BASE_URL``
                env variable, falling back to ``http://localhost:8880/v1``.
            api_key: Optional bearer token, for deployments behind an
                authenticating proxy. Kokoro-FastAPI itself requires none.
            http_session: Optional aiohttp session to use for requests.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            base_url=_resolve_base_url(base_url),
            api_key=api_key,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Kokoro"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        lang_code: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update the options for the TTS.

        Args:
            voice: Voice ID, e.g. ``af_heart``.
            model: Model name as exposed by the server.
            speed: Speech speed multiplier, between 0.25 and 4.0.
            lang_code: Language code for phonemization.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(model):
            self._opts.model = model
        if is_given(speed):
            self._opts.speed = speed
        if is_given(lang_code):
            self._opts.lang_code = lang_code

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text via a Kokoro-FastAPI server, streaming raw PCM chunks."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        json_data: dict = {
            "model": self._opts.model,
            "input": self._input_text,
            "voice": self._opts.voice,
            "response_format": "pcm",
            "stream": True,
            "speed": self._opts.speed,
        }
        if is_given(self._opts.lang_code):
            json_data["lang_code"] = self._opts.lang_code

        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/audio/speech",
                headers=_auth_headers(self._opts.api_key),
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
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
