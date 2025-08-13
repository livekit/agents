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
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from .models import STTModels


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str
    detect_language: bool
    prompt: NotGivenOr[str] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: STTModels | str = "voxtral-small-latest",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: Mistral | None = None,
    ):
        """
        Create a new instance of MistralAI STT.

        Args:
            language: The language code to use for transcription (e.g., "en" for English).
            detect_language: Whether to automatically detect the language.
            model: The MistralAI model to use for transcription (voxtral-small-latest or voxtral-mini-latest).
            prompt: Optional text prompt to guide the transcription.
            api_key: Your MistralAI API key. If not provided, will use the MISTRAL_API_KEY environment variable.
            client: Optional pre-configured MistralAI client instance.
        """

        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        if detect_language:
            language = ""

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
        )

        self._client = client or Mistral(
            api_key=api_key if is_given(api_key) else None,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the STT.

        Args:
            language: The language to transcribe in.
            detect_language: Whether to automatically detect the language.
            model: The model to use for transcription.
            prompt: Optional text prompt to guide the transcription.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(detect_language):
            self._opts.detect_language = detect_language
            self._opts.language = ""
        if is_given(prompt):
            self._opts.prompt = prompt

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
            prompt = self._opts.prompt if is_given(self._opts.prompt) else None

            # MistralAI transcription API call
            resp = await self._client.transcriptions.create(
                model=self._opts.model,
                file=("audio.wav", data, "audio/wav"),
                language=self._opts.language if self._opts.language else None,
                prompt=prompt,
            )

            sd = stt.SpeechData(text=resp.text, language=self._opts.language)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[sd],
            )

        except SDKError as e:
            if e.status_code in (408, 504):  # Request Timeout, Gateway Timeout
                raise APITimeoutError() from e
            else:
                raise APIStatusError(e.message, status_code=e.status_code, body=e.body) from e
        except Exception as e:
            raise APIConnectionError() from e
