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

import json
import os
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
    utils,
)

from google import genai  # type: ignore
from google.genai import types  # type: ignore

from .api_proto import LiveAPIModels

SAMPLE_RATE = 16000

SYSTEM_INSTRUCTIONS = """
You are an **Audio Transcriber**. Your task is to convert audio content into accurate and precise text.

**Guidelines:**

1. **Transcription Only:**
   - Transcribe spoken words exactly as they are.
   - Exclude any non-speech sounds (e.g., background noise, music).

2. **Response Format:**
   - Provide only the transcription without any additional text or explanations.
   - If the audio is unclear or inaudible, respond with: `...`

3. **Accuracy:**
   - Ensure the transcription is free from errors.
   - Maintain the original meaning and context of the speech.

4. **Clarity:**
   - Use proper punctuation and formatting to enhance readability.
   - Preserve the original speaker's intent and tone as much as possible.

**Do Not:**
- Add any explanations, comments, or additional information.
- Include timestamps, speaker labels, or annotations unless specified.
"""


@dataclass
class STTOptions:
    language: str
    detect_language: bool
    system_instructions: str
    model: LiveAPIModels


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = "en-US",
        detect_language: bool = True,
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        model: LiveAPIModels = "gemini-2.0-flash-exp",
    ):
        """
        Create a new instance of Google Realtime STT.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._config = STTOptions(
            language=language,
            model=model,
            system_instructions=system_instructions,
            detect_language=detect_language,
        )
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key=self._api_key,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            resp = await self._client.aio.models.generate_content(
                model=self._config.model,
                contents=types.Content(
                    parts=[
                        types.Part(
                            text=self._config.system_instructions,
                        ),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="audio/wav",
                                data=data,
                            )
                        ),
                    ],
                ),
                config=types.GenerationConfigDict(
                    response_mime_type="application/json",
                    response_schema={
                        "required": [
                            "transcribed_text",
                            "confidence_score",
                            "language",
                        ],
                        "properties": {
                            "transcribed_text": {"type": "STRING"},
                            "confidence_score": {"type": "NUMBER"},
                            "language": {"type": "STRING"},
                        },
                        "type": "OBJECT",
                    },
                ),
            )
            resp = json.loads(resp.text)
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.get("transcribed_text") or "",
                        language=resp.get("language") or self._config.language,
                    )
                ],
            )
        except Exception as e:
            raise APIConnectionError() from e
