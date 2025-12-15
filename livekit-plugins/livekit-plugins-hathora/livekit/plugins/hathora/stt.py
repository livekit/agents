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
from collections import deque
from dataclasses import dataclass
from typing import Optional, Literal, ByteString

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NotGivenOr, NOT_GIVEN

from .constants import (
    API_AUTH_HEADER,
    USER_AGENT,
)
from .log import logger
from .models import STTModels

@dataclass
class BaseSTTOptions:
    base_url: str
    model: STTModels
    api_key: Optional[str] = None
    sample_rate: Optional[int] = 16000

@dataclass
class ParakeetTDTSTTOptions(BaseSTTOptions):
    model: Literal['nvidia_parakeet_tdt_v3'] = 'nvidia_parakeet_tdt_v3'

class STT(stt.STT):
    def __init__(
        self,
        *,
        opts: ParakeetTDTSTTOptions,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._opts = opts
        self._opts.api_key = self._opts.api_key or os.environ.get("HATHORA_API_KEY")

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Hathora"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if isinstance(self._opts, ParakeetTDTSTTOptions):
            url = f"{self._opts.base_url}"

            url_query_params = []
            url_query_params.append(f"sample_rate={self._opts.sample_rate}")

            if len(url_query_params) > 0:
                url += "?" + "&".join(url_query_params)

            form_data = aiohttp.FormData()
            form_data.add_field("file", rtc.combine_audio_frames(buffer).to_wav_bytes(), filename="audio.wav", content_type="application/octet-stream")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        API_AUTH_HEADER: f"Bearer {self._opts.api_key}",
                        "User-Agent": USER_AGENT,
                    },
                    data=form_data,
                ) as resp:
                    response = await resp.json()

            if response and "text" in response:
                text = response["text"].strip()
                if text:
                    return stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=language or "en",
                                text=text,
                            )
                        ],
                    )

            raise APIStatusError("No text found in the response", status_code=400)

        raise NotImplementedError(f"Model {self._opts.model} is not supported")
