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

import io
import os
import wave
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .constants import (
    API_AUTH_HEADER,
    USER_AGENT,
)
from .utils import ConfigOption


def _decode_audio_payload(
    audio_bytes: bytes,
    *,
    fallback_sample_rate: int = 24000,
    fallback_channels: int = 1,
) -> tuple[bytes, int, int]:
    """Convert a WAV/PCM payload into raw PCM samples for TTSAudioRawFrame."""

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_reader:
            channels = wav_reader.getnchannels()
            sample_rate = wav_reader.getframerate()
            frames = wav_reader.readframes(wav_reader.getnframes())
            return frames, sample_rate, channels
    except (wave.Error, EOFError):
        # If the payload is already raw PCM, just pass it through.
        return audio_bytes, fallback_sample_rate, fallback_channels


class TTS(tts.TTS):
    """This service supports several different text-to-speech models hosted by Hathora.

    [Documentation](https://models.hathora.dev)
    """

    def __init__(
        self,
        *,
        model: str,
        voice: str | None = None,
        speed: float | None = None,
        model_config: list[ConfigOption] | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.models.hathora.dev/inference/v1/tts",
    ):
        """Initialize the Hathora TTS service.

        Args:
            model: Model to use; find available models
                [here](https://models.hathora.dev).
            voice: Voice to use for synthesis (if supported by model).
            speed: Speech speed multiplier (if supported by model).
            model_config: Some models support additional config, refer to
                [docs](https://models.hathora.dev) for each model to see
                what is supported.
            api_key: API key for authentication with the Hathora service;
                provision one [here](https://models.hathora.dev/tokens).
            base_url: Base API URL for the Hathora TTS service.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=24000,
            num_channels=1,
        )

        self._model = model
        self._voice = voice
        self._speed = speed
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

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using Hathora's unified TTS endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._text = input_text
        self._conn_options = conn_options

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        should_flush = False
        try:
            url = f"{self._tts._base_url}"

            payload: dict[str, Any] = {"model": self._tts._model, "text": self._text}

            if self._tts._voice is not None:
                payload["voice"] = self._tts._voice
            if self._tts._speed is not None:
                payload["speed"] = self._tts._speed
            if self._tts._model_config is not None:
                payload["model_config"] = [
                    {"name": option.name, "value": option.value}
                    for option in self._tts._model_config
                ]

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=24000,
                num_channels=1,
                mime_type="audio/pcm",
            )
            should_flush = True

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        API_AUTH_HEADER: f"Bearer {self._tts._api_key}",
                        "Accept": "application/octet-stream",
                        "User-Agent": USER_AGENT,
                    },
                    json=payload,
                ) as resp:
                    audio_data = await resp.read()

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self._tts.sample_rate,
            )

            output_emitter.push(pcm_audio)

        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=f"Hathora error: {e}", status_code=e.status, request_id=None, body=None
            ) from e
        finally:
            if should_flush:
                output_emitter.flush()
