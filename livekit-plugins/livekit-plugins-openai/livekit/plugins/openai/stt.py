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

from livekit import agents
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer

import openai

from .models import WhisperModels


@dataclass
class STTOptions:
    language: str
    detect_language: bool
    model: WhisperModels


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: WhisperModels = "whisper-1",
        api_key: str | None = None,
    ):
        super().__init__(streaming_supported=False)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self._client = openai.AsyncOpenAI(api_key=api_key)

        if detect_language:
            language = ""

        self._config = STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
        )

    def _sanitize_options(
        self,
        *,
        language: str | None = None,
    ) -> STTOptions:
        config = dataclasses.replace(self._config)
        config.language = language or config.language
        return config

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: str | None = None,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        resp = await self._client.audio.transcriptions.create(
            file=("a.wav", io_buffer),
            model=config.model,
            language=config.language,
            response_format="json",
        )
        return transcription_to_speech_event(resp, config.language)


def transcription_to_speech_event(transcription, language) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(text=transcription.text, language=language)],
    )
