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

- Transcribe verbatim; exclude non-speech sounds.
- Provide only transcription; no extra text or explanations.
- If audio is unclear, respond with: `...`
- Ensure error-free transcription, preserving meaning and context.
- Use proper punctuation and formatting.
- Do not add explanations, comments, or extra information.
- Do not include timestamps, speaker labels, or annotations unless specified.
"""


@dataclass
class STTOptions:
    language: str
    system_instructions: str
    model: LiveAPIModels


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = "en-US",
        vertexai: bool = False,
        project_id: str | None = None,
        location: str = "us-central1",
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        model: LiveAPIModels = "gemini-2.0-flash-exp",
    ):
        """
        Create a new instance of Google Realtime STT. you must provide either api_key or vertexai with project_id. api key and project id can be set via environment variables or via the arguments.
        Args:
            api_key (str, optional) : The API key to use for the API.
            vertexai(bool, optional) : Whether to use VertexAI.
                project_id(str, optional) : The project id to use for the vertex ai.
                location (str, optional) : The location to use for the vertex ai. defaults to us-central1
            system_instructions (str, optional) : custom system instructions to use for the transcription.
            language (str, optional) : The language of the audio. defaults to en-US
            model (LiveAPIModels, optional) : The model to use for the transcription. defaults to gemini-2.0-flash-exp
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._config = STTOptions(
            language=language, model=model, system_instructions=system_instructions
        )
        if vertexai:
            self._project_id = project_id or os.getenv("GOOGLE_PROJECT_ID")
            self._location = location or os.getenv("GOOGLE_LOCATION")
            if not self._project_id or not self._location:
                raise ValueError(
                    "Project and location are required for VertexAI either via project and location or GOOGLE_PROJECT_ID and GOOGLE_LOCATION environment variables"
                )
            self._api_key = None  # VertexAI does not require an API key
        else:
            self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"
                )
        self._client = genai.Client(
            api_key=self._api_key,
            vertexai=vertexai,
            project=self._project_id,
            location=self._location,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        conn_options: APIConnectOptions,
        language: str | None = None,
    ) -> stt.SpeechEvent:
        try:
            instructions = self._config.system_instructions
            if language:
                instructions += "The language of the audio is " + language
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            resp = await self._client.aio.models.generate_content(
                model=self._config.model,
                contents=[
                    types.Part(text=instructions),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="audio/wav",
                            data=data,
                        )
                    ),
                ],
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
