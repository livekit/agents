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
import io
import json
import os
import time
import wave

import aiohttp

from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.utils import AudioBuffer, merge_frames
from livekit.plugins.clova.constants import CLOVA_INPUT_SAMPLE_RATE

from .common import resample_audio
from .log import logger
from .models import ClovaSpeechAPIType, ClovaSttLanguages, clova_languages_mapping


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: ClovaSttLanguages | str = "en-US",
        secret: str | None = None,
        invoke_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        threshold: float = 0.5,
    ):
        """
        Create a new instance of Clova STT.

        ``secret`` and ``invoke_url`` must be set, either using arguments or by setting the
        ``CLOVA_STT_SECRET_KEY`` and ``CLOVA_STT_INVOKE_URL`` environmental variables, respectively.
        """

        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=True)
        )
        self._secret = secret or os.environ.get("CLOVA_STT_SECRET_KEY")
        self._invoke_url = invoke_url or os.environ.get("CLOVA_STT_INVOKE_URL")
        self._language = clova_languages_mapping.get(language, language)
        self._session = http_session
        if self._secret is None:
            raise ValueError(
                "Clova STT secret key is required. It should be set with env CLOVA_STT_SECRET_KEY"
            )
        self.threshold = threshold

    def update_options(self, *, language: str | None = None) -> None:
        self._language = (
            clova_languages_mapping.get(language, language) or self._language
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def url_builder(
        self, process_method: ClovaSpeechAPIType = "recognizer/upload"
    ) -> str:
        return f"{self._invoke_url}/{process_method}"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: ClovaSttLanguages | str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            url = self.url_builder()
            payload = json.dumps({"language": self._language, "completion": "sync"})

            buffer = merge_frames(buffer)
            buffer_bytes = resample_audio(
                buffer.data.tobytes(), buffer.sample_rate, CLOVA_INPUT_SAMPLE_RATE
            )

            io_buffer = io.BytesIO()
            with wave.open(io_buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(CLOVA_INPUT_SAMPLE_RATE)
                wav.writeframes(buffer_bytes)
            io_buffer.seek(0)

            headers = {"X-CLOVASPEECH-API-KEY": self._secret}
            form_data = aiohttp.FormData()
            form_data.add_field("params", payload)
            form_data.add_field(
                "media", io_buffer, filename="audio.wav", content_type="audio/wav"
            )
            start = time.time()
            async with self._ensure_session().post(
                url,
                data=form_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as response:
                response_data = await response.json()
                end = time.time()
                text = response_data.get("text")
                confidence = response_data.get("confidence")
                logger.info(f"{text} | {confidence} | total_seconds: {end - start}")
                if not text or "error" in response_data:
                    raise ValueError(f"Unexpected response: {response_data}")
                if confidence < self.threshold:
                    raise ValueError(
                        f"Confidence: {confidence} is bellow threshold {self.threshold}. Skipping."
                    )
                logger.info(f"final event: {response_data}")
                return self._transcription_to_speech_event(text=text)

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e

    def _transcription_to_speech_event(
        self,
        event_type: SpeechEventType = stt.SpeechEventType.INTERIM_TRANSCRIPT,
        text: str | None = None,
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[stt.SpeechData(text=text, language=self._language)],
        )
