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

import dataclasses
import os
from dataclasses import dataclass

import httpx
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.utils import AudioBuffer

import openai

from .models import GroqAudioModels, WhisperModels


@dataclass
class _STTOptions:
    language: str
    detect_language: bool
    model: WhisperModels | str


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: WhisperModels | str = "whisper-1",
        base_url: str | None = None,
        api_key: str | None = None,
        client: openai.AsyncClient | None = None,
    ):
        """
        Create a new instance of OpenAI STT.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        if detect_language:
            language = ""

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
        )

        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    def update_options(
        self,
        *,
        model: WhisperModels | GroqAudioModels | None = None,
        language: str | None = None,
    ) -> None:
        self._opts.model = model or self._opts.model
        self._opts.language = language or self._opts.language

    @staticmethod
    def with_groq(
        *,
        model: GroqAudioModels | str = "whisper-large-v3-turbo",
        api_key: str | None = None,
        base_url: str | None = "https://api.groq.com/openai/v1",
        client: openai.AsyncClient | None = None,
        language: str = "en",
        detect_language: bool = False,
    ) -> STT:
        """
        Create a new instance of Groq STT.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("Groq API key is required")

        return STT(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            language=language,
            detect_language=detect_language,
        )

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language=language)
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            resp = await self._client.audio.transcriptions.create(
                file=(
                    "file.wav",
                    data,
                    "audio/wav",
                ),
                model=self._opts.model,
                language=config.language,
                # verbose_json returns language and other details
                response_format="verbose_json",
                timeout=httpx.Timeout(30, connect=conn_options.timeout),
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text or "",
                        language=resp.language or config.language or "",
                    )
                ],
            )

        except openai.APITimeoutError:
            raise APITimeoutError()
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError() from e
