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

import os
import io
import wave
from typing import Optional
from livekit import agents
from livekit.agents.utils import AudioBuffer
from livekit.agents import stt
import openai
from .models import WhisperModels

WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


class STT(stt.STT):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(streaming_supported=False)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: str = "en",
        detect_language: bool = False,
        num_channels: int = 1,
        sample_rate: int = 16000,
        punctuate: bool = True,
        model: WhisperModels = "whisper-1",
    ) -> stt.SpeechEvent:
        buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        lang = ""
        if not detect_language:
            lang = language

        resp = await self._client.audio.transcriptions.create(
            file=("a.wav", io_buffer),
            model=model,
            language=lang,
            response_format="json",
        )
        return transcription_to_speech_event(resp)


def transcription_to_speech_event(transcription) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        is_final=True,
        alternatives=[stt.SpeechData(text=transcription.text, language="")],
    )
