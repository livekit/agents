# Copyright 2026 Samay AI (Quickdial)
# Licensed under the Apache License, Version 2.0
"""Quickdial STT for LiveKit Agents.

Non-streaming transcription over ``POST /v1/stt`` (multipart WAV) via
``_recognize_impl``. The STT declares ``streaming=False``; pair it with a VAD
(e.g. ``silero.VAD``) so the AgentSession segments speech and calls
``_recognize_impl`` per utterance, emitting a FINAL_TRANSCRIPT (no interim
results yet). This mirrors the OpenAI Whisper STT plugin.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .models import STTLanguages

DEFAULT_BASE_URL = "https://api.quickdial.ai"
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


@dataclass
class _STTOptions:
    language: str
    sample_rate: int
    base_url: str
    api_key: str
    params: dict | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: STTLanguages | str = "en",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: int = SAMPLE_RATE,
        params: NotGivenOr[dict] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        # Quickdial returns a transcript per utterance (no interim results), so we run
        # as a NON-streaming STT: the AgentSession's VAD segments speech and calls
        # _recognize_impl (POST /v1/stt) per utterance.
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        key = api_key if is_given(api_key) else os.environ.get("QUICKDIAL_API_KEY", "")
        if not key:
            raise ValueError("Quickdial API key required — pass api_key= or set QUICKDIAL_API_KEY")
        self._opts = _STTOptions(
            language=language,
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

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("audio", wav, filename="audio.wav", content_type="audio/wav")
        cfg = dict(self._opts.params or {})
        cfg["language"] = language if is_given(language) else self._opts.language
        form.add_field("params", json.dumps(cfg))
        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/v1/stt",
                headers={"Authorization": f"Bearer {self._opts.api_key}"},
                data=form,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return _to_speech_event(data, self._opts.language)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status) from e
        except Exception as e:
            raise APIConnectionError() from e


def _to_speech_event(data: dict, language: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=data.get("language", language),
                text=data.get("text", ""),
            )
        ],
    )
