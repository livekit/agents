# Copyright 2025 LiveKit, Inc.
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
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx

from livekit.agents import tts, utils
from livekit.agents._exceptions import APIConnectionError, APITimeoutError
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)

AUDIO_ENCODING = "LINEAR16"
INWORLD_API_BASE_URL = "https://api.inworld.ai/"
MIME_TYPE = "audio/pcm"
MODEL_ID = "inworld-tts-1"
NUM_CHANNELS = 1
PITCH = 0.0
SAMPLE_RATE = 24000
SPEED = 1.0
TEMPERATURE = 0.8
WAV_HEADER_SIZE = 44
VOICE_ID = "Olivia"


@dataclass
class _TTSOptions:
    modelId: str | None
    voice: str | None
    pitch: float | None
    speed: float | None
    sampleRateHertz: int | None
    temperature: float | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        pitch: float = PITCH,
        speed: float = SPEED,
        sample_rate: int = SAMPLE_RATE,
        temperature: float = TEMPERATURE,
        base_url: str = INWORLD_API_BASE_URL,
        auth_type: Literal["basic", "bearer"] = "basic",
    ) -> None:
        """
        Create a new instance of Inworld TTS.

        Args:
            api_key (str, optional): The Inworld API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            model (str, optional): The Inworld model to use. Defaults to "inworld-tts-1".
            voice (str, optional): The voice to use. Defaults to "Olivia".
            pitch (float, optional): The pitch of the voice. Defaults to 0.0.
            speed (float, optional): The speed of the voice. Defaults to 1.0.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            temperature (float, optional): The temperature of the voice. Defaults to 0.8.
            base_url (str, optional): The base URL for the Inworld API.
            auth_type (Literal["basic", "bearer"], optional): The authentication type to use.
                Defaults to "basic".
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError("Inworld API key required. Set INWORLD_API_KEY or provide api_key.")

        auth_prefix = "Basic" if auth_type.lower() == "basic" else "Bearer"

        self._opts = _TTSOptions(
            modelId=model,
            voice=voice,
            pitch=pitch,
            speed=speed,
            sampleRateHertz=sample_rate,
            temperature=temperature,
        )

        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"{auth_prefix} {api_key}",
            },
            timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50, max_keepalive_connections=50, keepalive_expiry=120
            ),
        )

    def update_options(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        pitch: float | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Update the TTS configuration options.

        Args:
            model (str, optional): The Inworld model to use.
            voice (str, optional): The voice to use.
            pitch (float, optional): The pitch of the voice.
            speed (float, optional): The speed of the voice.
            sample_rate (int, optional): The audio sample rate in Hz.
            temperature (float, optional): The temperature of the voice.
        """
        if model is not None:
            self._opts.modelId = model
        if voice is not None:
            self._opts.voice = voice
        if pitch is not None:
            self._opts.pitch = pitch
        if speed is not None:
            self._opts.speed = speed
        if sample_rate is not None:
            self._opts.sampleRateHertz = sample_rate
        if temperature is not None:
            self._opts.temperature = temperature

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sampleRateHertz or SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type=MIME_TYPE,
        )

        try:
            async with self._create_synthesis() as synthesis:
                async for chunk in synthesis.aiter_lines():
                    if chunk:
                        chunk_data = json.loads(chunk)
                        if chunk_data and (chunk_result := chunk_data.get("result")):
                            if chunk_result and (audio_content := chunk_result.get("audioContent")):
                                audio_data = base64.b64decode(audio_content)
                                audio_data = _strip_wav_header(audio_data)
                                output_emitter.push(audio_data)
            output_emitter.flush()
        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            output_emitter.flush()

    def _create_synthesis(self) -> AbstractAsyncContextManager[httpx.Response]:
        return self._tts._http_client.stream(
            "POST",
            "/tts/v1/voice:stream",
            json=_generate_request(self._opts, self._input_text),
            timeout=httpx.Timeout(
                timeout=self._conn_options.timeout,
            ),
        )


def _strip_wav_header(audio_data: bytes) -> bytes:
    if len(audio_data) <= WAV_HEADER_SIZE:
        return audio_data
    elif audio_data.startswith(b"RIFF") and audio_data[8:12] == b"WAVE":
        return audio_data[WAV_HEADER_SIZE:]
    else:
        return audio_data


def _generate_request(
    opts: _TTSOptions, input: str
) -> dict[str, str | dict[str, str | float | int]]:
    data: dict[str, Any] = {
        "text": input,
        "voiceId": opts.voice or VOICE_ID,
        "modelId": opts.modelId or MODEL_ID,
        "audioConfig": {
            "audioEncoding": AUDIO_ENCODING,
            "pitch": opts.pitch or PITCH,
            "sampleRateHertz": opts.sampleRateHertz or SAMPLE_RATE,
            "speakingRate": opts.speed or SPEED,
        },
        "temperature": opts.temperature or TEMPERATURE,
    }

    return data
