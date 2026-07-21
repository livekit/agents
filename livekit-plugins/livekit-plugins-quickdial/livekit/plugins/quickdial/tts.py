# Copyright 2026 Samay AI (Quickdial)
# Licensed under the Apache License, Version 2.0
"""Quickdial TTS for LiveKit Agents.

Non-streaming REST synthesis: ``ChunkedStream`` does a one-shot ``POST /v1/tts``
(streaming WAV) via ``synthesize()``, emitting 16-bit PCM mono @ 24 kHz. The TTS
declares ``streaming=False``; the ``AgentSession`` wraps it in a ``StreamAdapter``
and synthesizes per sentence, so the first words play before the reply finishes
(the same pattern the OpenAI and Hume plugins use).

Auth is a Bearer API key. Get one at https://web.quickdial.ai.
"""

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

from .models import TTSVoices

DEFAULT_BASE_URL = "https://api.quickdial.ai"
DEFAULT_VOICE = "alba"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    voice: str
    sample_rate: int
    base_url: str
    api_key: str
    params: dict | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: TTSVoices | str = DEFAULT_VOICE,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: int = SAMPLE_RATE,
        params: NotGivenOr[dict] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Quickdial TTS.

        Args:
            voice: voice name (see ``GET /v1/voices``). Defaults to ``alba``.
            api_key: Quickdial API key; falls back to ``QUICKDIAL_API_KEY``.
            base_url: API base, default ``https://api.quickdial.ai``.
            sample_rate: output sample rate (Quickdial streams 24 kHz).
            params: optional Pocket-TTS knobs (``temperature``, ``speed``, …).
        """
        # Non-streaming: with streaming=False the AgentSession wraps this in a
        # StreamAdapter and synthesizes per sentence over POST /v1/tts, giving
        # continuous audio with low time-to-first-audio.
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        key = api_key if is_given(api_key) else os.environ.get("QUICKDIAL_API_KEY", "")
        if not key:
            raise ValueError("Quickdial API key required — pass api_key= or set QUICKDIAL_API_KEY")

        self._opts = _TTSOptions(
            voice=voice,
            sample_rate=sample_rate,
            base_url=base_url.rstrip("/"),
            api_key=key,
            params=params if is_given(params) else None,
        )
        self._session = http_session

    @property
    def provider(self) -> str:
        return "Quickdial"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self, *, voice: NotGivenOr[str] = NOT_GIVEN, params: NotGivenOr[dict] = NOT_GIVEN
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(params):
            self._opts.params = params

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """One-shot synthesis over ``POST /v1/tts`` (streaming WAV)."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body: dict = {"text": self._input_text, "voice": self._opts.voice, "format": "wav"}
        if self._opts.params:
            body["params"] = self._opts.params
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/v1/tts",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",
                )
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e
