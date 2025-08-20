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
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, http_context, is_given

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    language_code: str = "en"


class STT(stt.STT):
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        language_code: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code for the STT model. Optional.
        """  # noqa: E501
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=True))

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or "
                "set ELEVEN_API_KEY environmental variable"
            )
        self._opts = _STTOptions(
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            language_code=language_code if is_given(language_code) else "en",
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if is_given(language):
            self._opts.language_code = language

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/x-wav")
        form.add_field("model_id", "scribe_v1")
        form.add_field("language_code", self._opts.language_code)

        try:
            async with self._ensure_session().post(
                f"{API_BASE_URL_V1}/speech-to-text",
                data=form,
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
            ) as response:
                response_json = await response.json()
                extracted_text = response_json.get("text")
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

        return self._transcription_to_speech_event(text=extracted_text)

    def _transcription_to_speech_event(self, text: str) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=self._opts.language_code)],
        )
