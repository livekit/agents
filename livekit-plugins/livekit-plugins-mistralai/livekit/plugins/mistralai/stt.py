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

import os
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from .models import STTModels


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str | None = "en",
        model: STTModels | str = "voxtral-mini-latest",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: Mistral | None = None,
    ):
        """
        Create a new instance of MistralAI STT.

        Args:
            language: The language code to use for transcription (e.g., "en" for English). Segment timestamps will only be available if set to None.
            model: The MistralAI model to use for transcription, default is voxtral-mini-latest.
            api_key: Your MistralAI API key. If not provided, will use the MISTRAL_API_KEY environment variable.
            client: Optional pre-configured MistralAI client instance.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript=False,
            )
        )
        self._opts = _STTOptions(
            language=language,
            model=model,
        )

        mistral_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MistralAI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._client = client or Mistral(api_key=mistral_api_key)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the STT.

        Args:
            language: The language to transcribe in.
            detect_language: Whether to automatically detect the language.
            model: The model to use for transcription.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            if is_given(language):
                self._opts.language = language
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            # MistralAI transcription API call
            resp = await self._client.audio.transcriptions.complete_async(
                model=self._opts.model,
                file={"content": data, "file_name": "audio.wav"},
                language=self._opts.language if self._opts.language else None,
                timestamp_granularities=None if self._opts.language else ["segment"],
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=resp.text,
                        language=self._opts.language if self._opts.language else "",
                        start_time=resp.segments[0].start if resp.segments else 0,
                        end_time=resp.segments[-1].end if resp.segments else 0,
                        words=[
                            TimedString(
                                text=segment.text,
                                start_time=segment.start,
                                end_time=segment.end,
                            )
                            for segment in resp.segments
                        ]
                        if resp.segments
                        else None,
                    ),
                ],
            )

        except SDKError as e:
            if e.status_code in (408, 504):  # Request Timeout, Gateway Timeout
                raise APITimeoutError() from e
            else:
                raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
