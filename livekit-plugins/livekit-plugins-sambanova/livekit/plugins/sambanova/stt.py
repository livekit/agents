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
from dataclasses import dataclass
from typing import Any

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
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .models import STTModels


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str | None
    detect_language: bool
    prompt: NotGivenOr[str]
    temperature: NotGivenOr[float]
    base_url: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "Whisper-Large-v3",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = "https://api.sambanova.ai/v1",
        language: str = "en",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        detect_language: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of SambaNova STT.

        ``api_key`` must be set to your SambaNova API key, either using the argument or by
        setting the ``SAMBANOVA_API_KEY`` environmental variable.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript=False,
            )
        )

        self._api_key = _get_api_key(api_key)
        self._session = http_session
        self._opts = _STTOptions(
            model=model,
            language=None if detect_language else language,
            detect_language=detect_language,
            prompt=prompt,
            temperature=temperature,
            base_url=base_url,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SambaNova"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(detect_language):
            self._opts.detect_language = detect_language
            if detect_language:
                self._opts.language = None
        if is_given(language):
            self._opts.language = language
            self._opts.detect_language = False
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(base_url):
            self._opts.base_url = base_url

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        transcribe_language = self._opts.language
        if is_given(language):
            transcribe_language = language

        data = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("file", data, filename="audio.wav", content_type="audio/wav")
        form.add_field("model", self._opts.model)
        form.add_field("response_format", "json")
        if transcribe_language:
            form.add_field("language", transcribe_language)
        if is_given(self._opts.prompt):
            form.add_field("prompt", self._opts.prompt)
        if is_given(self._opts.temperature):
            form.add_field("temperature", str(self._opts.temperature))

        try:
            async with self._ensure_session().post(
                self._transcriptions_url(),
                data=form,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=conn_options.timeout),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise APIStatusError(
                        message=_error_message(body),
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        body=body,
                    )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(message=e.message, status_code=e.status, request_id=None) from e
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

        text = _get_text(body)
        segments = _get_segments(body)
        words = _to_words(segments)
        detected_language = _get_language(body) or transcribe_language or ""
        start_time = segments[0].get("start", 0.0) if segments else 0.0
        end_time = segments[-1].get("end", 0.0) if segments else 0.0

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=detected_language,
                    text=text,
                    start_time=float(start_time) if start_time else 0.0,
                    end_time=float(end_time) if end_time else 0.0,
                    words=words,
                )
            ],
        )

    def _transcriptions_url(self) -> str:
        base = self._opts.base_url.rstrip("/")
        if base.endswith("/audio/transcriptions"):
            return base
        return f"{base}/audio/transcriptions"


def _get_api_key(key: NotGivenOr[str]) -> str:
    sambanova_api_key = key if is_given(key) else os.environ.get("SAMBANOVA_API_KEY")
    if not sambanova_api_key:
        raise ValueError(
            "SAMBANOVA_API_KEY is required, either as argument or set SAMBANOVA_API_KEY"
            " environmental variable"
        )
    return sambanova_api_key


def _get_text(body: Any) -> str:
    if isinstance(body, dict):
        text = body.get("text")
        if isinstance(text, str):
            return text
    return ""


def _get_language(body: Any) -> str | None:
    if isinstance(body, dict):
        language = body.get("language")
        if isinstance(language, str):
            return language
    return None


def _get_segments(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    segments = body.get("segments")
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def _to_words(segments: list[dict[str, Any]]) -> list[TimedString] | None:
    if not segments:
        return None

    words: list[TimedString] = []
    for segment in segments:
        text = segment.get("text")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(text, str):
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        words.append(TimedString(text=text, start_time=float(start), end_time=float(end)))

    return words or None


def _error_message(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str):
                return message
        if isinstance(err, str):
            return err
    return "SambaNova API request failed"
