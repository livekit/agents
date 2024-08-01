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
import io
import os
import wave
from dataclasses import dataclass

import aiohttp
from livekit import agents
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer

from .models import WhisperModels
from .utils import get_base_url


@dataclass
class _STTOptions:
    language: str
    detect_language: bool
    model: WhisperModels
    api_key: str
    endpoint: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: WhisperModels = "whisper-1",
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        if detect_language:
            language = ""

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
            api_key=api_key,
            endpoint=os.path.join(get_base_url(base_url), "audio/transcriptions"),
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        return config

    async def recognize(
        self, buffer: AudioBuffer, *, language: str | None = None
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        form = aiohttp.FormData()
        form.add_field("file", io_buffer.getvalue(), filename="my_file.wav")
        form.add_field("model", config.model)

        if config.language:
            form.add_field("language", config.language)

        form.add_field("response_format", "json")

        async with self._ensure_session().post(
            self._opts.endpoint,
            headers={"Authorization": f"Bearer {config.api_key}"},
            data=form,
        ) as resp:
            data = await resp.json()
            if "text" not in data or "error" in data:
                raise ValueError(f"Unexpected response: {data}")

            return _transcription_to_speech_event(data, config.language)


def _transcription_to_speech_event(transcription: dict, language) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(text=transcription["text"], language=language)],
    )
