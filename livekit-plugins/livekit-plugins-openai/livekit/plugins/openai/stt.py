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
from livekit.agents import stt
from dataclasses import dataclass
import openai
from .models import WhisperModels

WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1


@dataclass
class RecognizeOptions(stt.RecognizeOptions):
    model: WhisperModels = "whisper-1"

    # https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
    language: str = "en"


class STT(stt.STT):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(streaming_supported=False)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def recognize(
        self,
        buffer: agents.AudioBuffer,
        opts: stt.RecognizeOptions = stt.RecognizeOptions(),
    ) -> stt.SpeechEvent:
        buffer = agents.utils.merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        lang = ""
        if not opts.detect_language:
            lang = getattr(opts, "language", "en")

        resp = await self._client.audio.transcriptions.create(
            file=("a.wav", io_buffer),
            model=getattr(opts, "model", "whisper-1"),
            language=lang,
            response_format="json",
        )
        return transcription_to_speech_event(opts, resp)


def transcription_to_speech_event(
    opts: RecognizeOptions, transcription
) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        is_final=True,
        alternatives=[stt.SpeechData(text=transcription.text, language="")],
    )
