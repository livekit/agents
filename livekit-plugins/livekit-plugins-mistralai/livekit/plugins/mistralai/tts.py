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
import json
import os
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSModels

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

DEFAULT_MODEL = "voxtral-tts-2603"
DEFAULT_BASE_URL = "https://api.mistral.ai"

RESPONSE_FORMATS = Literal["mp3", "opus", "flac", "wav", "pcm"]


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str | None
    response_format: RESPONSE_FORMATS


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[RESPONSE_FORMATS] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of MistralAI TTS.

        Args:
            model: The MistralAI TTS model to use, default is voxtral-tts-2603.
            voice: The voice ID to use for speech generation (preset or custom).
            api_key: Your MistralAI API key. If not provided, will use the
                MISTRAL_API_KEY environment variable.
            base_url: Custom base URL for the API. Defaults to https://api.mistral.ai.
            response_format: Output audio format. Defaults to mp3.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            voice=voice if is_given(voice) else None,
            response_format=response_format if is_given(response_format) else "mp3",
        )

        resolved_api_key = api_key if is_given(api_key) else os.environ.get("MISTRAL_API_KEY")
        if not resolved_api_key:
            raise ValueError("MistralAI API key is required. Set MISTRAL_API_KEY or pass api_key")
        self._api_key = resolved_api_key
        self._base_url = base_url if is_given(base_url) else DEFAULT_BASE_URL

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
            ),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MistralAI"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._http_client.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        headers = {
            "Authorization": f"Bearer {self._tts._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload: dict = {
            "model": self._opts.model,
            "input": self.input_text,
            "stream": True,
            "response_format": self._opts.response_format,
        }
        if self._opts.voice:
            payload["voice_id"] = self._opts.voice

        try:
            async with self._tts._http_client.stream(
                "POST",
                f"{self._tts._base_url}/v1/audio/speech",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(30, connect=self._conn_options.timeout),
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise APIStatusError(
                        body.decode("utf-8", errors="replace"),
                        status_code=response.status_code,
                        body=body,
                    )

                output_emitter.initialize(
                    request_id="",
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.response_format}",
                )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "speech.audio.delta":
                        audio_b64 = event.get("audio_data", "")
                        if audio_b64:
                            output_emitter.push(base64.b64decode(audio_b64))

                    elif event_type == "speech.audio.done":
                        usage = event.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        if input_tokens or output_tokens:
                            self._set_token_usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )

            output_emitter.flush()

        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except APITimeoutError:
            raise
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
