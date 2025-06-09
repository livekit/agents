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

"""Text-to-Speech implementation for Sarvam.ai

This module provides a TTS implementation that uses the Sarvam.ai API.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Literal

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger

SARVAM_TTS_BASE_URL = "https://api.sarvam.ai/text-to-speech"

# Sarvam TTS specific models and speakers
SarvamTTSModels = Literal["bulbul:v2"]

# Supported languages in BCP-47 format
SarvamTTSLanguages = Literal[
    "bn-IN",  # Bengali
    "en-IN",  # English (India)
    "gu-IN",  # Gujarati
    "hi-IN",  # Hindi
    "kn-IN",  # Kannada
    "ml-IN",  # Malayalam
    "mr-IN",  # Marathi
    "od-IN",  # Odia
    "pa-IN",  # Punjabi
    "ta-IN",  # Tamil
    "te-IN",  # Telugu
]

SarvamTTSSpeakers = Literal[
    # bulbul:v2 Female (lowercase)
    "anushka",
    "manisha",
    "vidya",
    "arya",
    # bulbul:v2 Male (lowercase)
    "abhilash",
    "karun",
    "hitesh",
]

# Model-Speaker compatibility mapping
MODEL_SPEAKER_COMPATIBILITY = {
    "bulbul:v2": {
        "female": ["anushka", "manisha", "vidya", "arya"],
        "male": ["abhilash", "karun", "hitesh"],
        "all": ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"],
    },
}


def validate_model_speaker_compatibility(model: str, speaker: str) -> bool:
    """Validate that the speaker is compatible with the model version."""
    if model not in MODEL_SPEAKER_COMPATIBILITY:
        logger.warning(f"Unknown model '{model}', skipping compatibility check")
        return True

    compatible_speakers = MODEL_SPEAKER_COMPATIBILITY[model]["all"]
    if speaker.lower() not in compatible_speakers:
        logger.error(
            f"Speaker '{speaker}' is not compatible with model '{model}'. "
            f"Compatible speakers for {model}: {', '.join(compatible_speakers)}"
        )
        return False
    return True


@dataclass
class SarvamTTSOptions:
    """Options for the Sarvam.ai TTS service.

    Args:
        target_language_code: BCP-47 language code for supported Indian languages
        api_key: Sarvam.ai API key
        text: The text to synthesize (will be provided by stream adapter)
        speaker: Voice to use for synthesis
        pitch: Voice pitch adjustment (-20.0 to 20.0)
        pace: Speech rate multiplier (0.5 to 2.0)
        loudness: Volume multiplier (0.5 to 2.0)
        speech_sample_rate: Audio sample rate (8000, 16000, 22050, or 24000)
        enable_preprocessing: Whether to use text preprocessing
        model: The Sarvam TTS model to use
        base_url: API endpoint URL
    """

    target_language_code: SarvamTTSLanguages | str  # BCP-47 for supported Indian languages
    api_key: str  # Sarvam.ai API key
    text: str | None = None  # Will be provided by the stream adapter
    speaker: SarvamTTSSpeakers | str = "manisha"  # Default speaker compatible with v2
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    speech_sample_rate: int = 22050  # Default 22050 Hz
    enable_preprocessing: bool = False
    model: SarvamTTSModels | str = "bulbul:v2"  # Default to v2 as it has more recent speakers
    base_url: str = SARVAM_TTS_BASE_URL


class TTS(tts.TTS):
    """Sarvam.ai Text-to-Speech implementation.

    This class provides text-to-speech functionality using the Sarvam.ai API.
    Sarvam.ai specializes in high-quality TTS for Indian languages.

    Args:
        target_language_code: BCP-47 language code for supported Indian languages
        model: Sarvam TTS model to use (only bulbul:v2 supported)
        speaker: Voice to use for synthesis
        speech_sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels (Sarvam outputs mono)
        pitch: Voice pitch adjustment (-20.0 to 20.0)
        pace: Speech rate multiplier (0.5 to 2.0)
        loudness: Volume multiplier (0.5 to 2.0)
        enable_preprocessing: Whether to use text preprocessing
        api_key: Sarvam.ai API key (required)
        base_url: API endpoint URL
        http_session: Optional aiohttp session to use
    """

    def __init__(
        self,
        *,
        target_language_code: SarvamTTSLanguages | str,
        model: SarvamTTSModels | str = "bulbul:v2",
        speaker: SarvamTTSSpeakers | str = "manisha",
        speech_sample_rate: int = 22050,
        num_channels: int = 1,  # Sarvam output is mono WAV
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
        enable_preprocessing: bool = False,
        api_key: str,
        base_url: str = SARVAM_TTS_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=speech_sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. Provide it directly or set SARVAM_API_KEY env var."
            )

        # Validate model-speaker compatibility
        if not validate_model_speaker_compatibility(model, speaker):
            compatible_speakers = MODEL_SPEAKER_COMPATIBILITY.get(model, {}).get("all", [])
            raise ValueError(
                f"Speaker '{speaker}' is not compatible with model '{model}'. "
                f"Please choose a compatible speaker from: {', '.join(compatible_speakers)}"
            )

        self._opts = SarvamTTSOptions(
            target_language_code=target_language_code,
            model=model,
            speaker=speaker,
            speech_sample_rate=speech_sample_rate,
            pitch=pitch,
            pace=pace,
            loudness=loudness,
            enable_preprocessing=enable_preprocessing,
            api_key=self._api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._logger = logger.getChild(self.__class__.__name__)

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    # Implement the abstract synthesize method
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> ChunkedStream:
        """Synthesize text to audio using Sarvam.ai TTS API."""
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the Sarvam.ai TTS request and emit audio via the output emitter."""
        payload = {
            "target_language_code": self._tts._opts.target_language_code,
            "text": self._input_text,
            "speaker": self._tts._opts.speaker,
            "pitch": self._tts._opts.pitch,
            "pace": self._tts._opts.pace,
            "loudness": self._tts._opts.loudness,
            "speech_sample_rate": self._tts._opts.speech_sample_rate,
            "enable_preprocessing": self._tts._opts.enable_preprocessing,
            "model": self._tts._opts.model,
        }
        headers = {
            "api-subscription-key": self._tts._opts.api_key,
            "Content-Type": "application/json",
        }
        try:
            async with self._tts._ensure_session().post(
                url=self._tts._opts.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    self._tts._logger.error(f"Sarvam TTS API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Sarvam TTS API Error: {error_text}", status_code=res.status
                    )

                response_json = await res.json()
                request_id = response_json.get("request_id", "")
                audios = response_json.get("audios", [])
                if not audios or not isinstance(audios, list):
                    raise APIConnectionError("Sarvam TTS API response invalid: no audio data")

                wav_bytes = base64.b64decode(audios[0])
                output_emitter.initialize(
                    request_id=request_id or "unknown",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/wav",
                )
                output_emitter.push(wav_bytes)
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Sarvam TTS API request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Sarvam TTS API connection error: {e}") from e
