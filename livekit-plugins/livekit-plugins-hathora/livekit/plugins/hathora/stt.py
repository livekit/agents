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

import base64
import os
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .constants import (
    API_AUTH_HEADER,
    USER_AGENT,
)
from .utils import ConfigOption


class STT(stt.STT):
    """This service supports several different speech-to-text models hosted by Hathora.

    [Documentation](https://models.hathora.dev)
    """

    def __init__(
        self,
        *,
        model: str,
        language: str | None = None,
        model_config: list[ConfigOption] | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.models.hathora.dev/inference/v1/stt",
    ):
        """Initialize the Hathora STT service.

        Args:
            model: Model to use; find available models
                [here](https://models.hathora.dev).
            language: Language code (if supported by model).
            model_config: Some models support additional config, refer to
                [docs](https://models.hathora.dev) for each model to see
                what is supported.
            api_key: API key for authentication with the Hathora service;
                provision one [here](https://models.hathora.dev/tokens).
            base_url: Base API URL for the Hathora STT service.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._model = model
        self._language = language
        self._model_config = model_config
        self._api_key = api_key or os.environ.get("HATHORA_API_KEY")
        self._base_url = base_url

    @property
    def model(self) -> str:
        """Get the model name/identifier for this TTS instance.

        Returns:
            The model name.
        """
        return self._model

    @property
    def provider(self) -> str:
        """Get the provider name/identifier for this TTS instance.

        Returns:
            "Hathora"
        """
        return "Hathora"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        url = f"{self._base_url}"

        payload: dict[str, Any] = {
            "model": self._model,
        }

        if self._language is not None:
            payload["language"] = self._language
        elif language is not NOT_GIVEN:
            payload["language"] = language

        if self._model_config is not None:
            payload["model_config"] = [
                {"name": option.name, "value": option.value} for option in self._model_config
            ]

        bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        base64_audio = base64.b64encode(bytes).decode("utf-8")
        payload["audio"] = base64_audio

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={
                    API_AUTH_HEADER: f"Bearer {self._api_key}",
                    "User-Agent": USER_AGENT,
                },
                json=payload,
            ) as resp:
                response = await resp.json()

        if response and "text" in response:
            text = response["text"].strip()
            returned_language = response.get("language", None)
            if text:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=returned_language or language or "en",
                            text=text,
                        )
                    ],
                )

        raise APIStatusError("No text found in the response", status_code=400)
