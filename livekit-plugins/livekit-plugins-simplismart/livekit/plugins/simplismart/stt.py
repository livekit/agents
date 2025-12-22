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

"""Speech-to-Text implementation for SimpliSmart

This module provides an STT implementation that uses the SimpliSmart API.
"""

import asyncio
import base64
import os
from typing import Any, Literal

import aiohttp
from pydantic import BaseModel

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, rtc

from .log import logger


class SimplismartSTTOptions(BaseModel):
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    without_timestamps: bool = True
    vad_model: Literal["silero", "frame"] = "frame"
    vad_filter: bool = True
    model: str | None = "openai/whisper-large-v3-turbo"
    word_timestamps: bool = False
    vad_onset: float | None = 0.5
    vad_offset: float | None = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = 30
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    diarization: bool = False
    initial_prompt: str | None = None
    hotwords: str | None = None
    num_speakers: int = 0
    compression_ratio_threshold: float | None = 2.4
    beam_size: int = 4
    temperature: float = 0.0
    multilingual: bool = False
    max_tokens: float | None = 400
    log_prob_threshold: float | None = -1.0
    length_penalty: int = 1
    repetition_penalty: float = 1.01
    suppress_tokens: list[int] = [-1]
    strict_hallucination_reduction: bool = False


class STT(stt.STT):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | SimplismartSTTOptions | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript="word",
            )
        )

        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        if params is None:
            params = SimplismartSTTOptions()

        if isinstance(params, SimplismartSTTOptions):
            self._opts = params
            self._model = params.model
        else:
            self._opts = SimplismartSTTOptions(**params)

        self._base_url = base_url
        self._logger = logger.getChild(self.__class__.__name__)
        self._session = http_session


    @property
    def provider(self) -> str:
        return "Simplismart"

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
        language = self._opts.language if isinstance(language, type(NOT_GIVEN)) else language
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = self._opts.model_dump()

        payload["audio_data"] = audio_b64
        payload["language"] = language
        payload["model"] = self._model

        try:
            async with self._ensure_session().post(
                self._base_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    self._logger.error(f"Simplismart API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Simplismart API Error: {error_text}",
                        status_code=res.status,
                    )

                response_json = await res.json()

                detected_language = response_json["info"]["language"]

                start_time = response_json["timestamps"][0][0]
                end_time = response_json["timestamps"][-1][1]
                request_id = response_json.get("request_id", "")
                text = "".join(response_json["transcription"])

                alternatives = [
                    stt.SpeechData(
                        language=detected_language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=1.0,
                    ),
                ]

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )
        except asyncio.TimeoutError as e:
            self._logger.error(f"Simplismart API timeout: {e}")
            raise APITimeoutError("Simplismart API request timed out") from e
        except aiohttp.ClientError as e:
            self._logger.error(f"Simplismart API client error: {e}")
            raise APIConnectionError(f"Simplismart API connection error: {e}") from e
        except Exception as e:
            self._logger.error(f"Error during Simplismart STT processing: {e}")
            raise APIConnectionError(f"Unexpected error in Simplismart STT: {e}") from e
