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
import io
import os
import wave
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Literal, Optional

import aiohttp

from livekit import rtc
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
SarvamTTSModels = Literal["bulbul:v1", "bulbul:v2"]
SarvamTTSSpeakers = Literal[
    # bulbul:v1 Female (lowercase)
    "diya",
    "maya",
    "meera",
    "pavithra",
    "maitreyi",
    "misha",
    # bulbul:v1 Male (lowercase)
    "amol",
    "arjun",
    "amartya",
    "arvind",
    "neel",
    "vian",
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
    "bulbul:v1": {
        "female": ["diya", "maya", "meera", "pavithra", "maitreyi", "misha"],
        "male": ["amol", "arjun", "amartya", "arvind", "neel", "vian"],
        "all": ["diya", "maya", "meera", "pavithra", "maitreyi", "misha", 
                "amol", "arjun", "amartya", "arvind", "neel", "vian"]
    },
    "bulbul:v2": {
        "female": ["anushka", "manisha", "vidya", "arya"],
        "male": ["abhilash", "karun", "hitesh"],
        "all": ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"]
    }
}

def validate_model_speaker_compatibility(model: str, speaker: str) -> bool:
    """Validate that the speaker is compatible with the model version."""
    if model not in MODEL_SPEAKER_COMPATIBILITY:
        logger.warning(f"Unknown model '{model}', skipping compatibility check")
        return True
    
    compatible_speakers = MODEL_SPEAKER_COMPATIBILITY[model]["all"]
    if speaker.lower() not in compatible_speakers:
        logger.error(f"Speaker '{speaker}' is not compatible with model '{model}'. "
                    f"Compatible speakers for {model}: {', '.join(compatible_speakers)}")
        return False
    return True

@dataclass
class SarvamTTSOptions:
    """Options for the Sarvam.ai TTS service.

    Args:
        target_language_code: BCP-47 language code, e.g., "hi-IN"
        text: The text to synthesize (will be provided by stream adapter)
        speaker: Voice to use for synthesis
        pitch: Voice pitch adjustment (-20.0 to 20.0)
        pace: Speech rate multiplier (0.5 to 2.0)
        loudness: Volume multiplier (0.5 to 2.0)
        speech_sample_rate: Audio sample rate (8000, 16000, 22050, or 24000)
        enable_preprocessing: Whether to use text preprocessing
        model: The Sarvam TTS model to use
        api_key: Sarvam.ai API key
        base_url: API endpoint URL
    """

    target_language_code: str  # BCP-47, e.g., "hi-IN"
    text: str | None = None  # Will be provided by the stream adapter
    speaker: SarvamTTSSpeakers | str = "manisha"  # Default speaker compatible with v2
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    speech_sample_rate: int = 22050  # Default 22050 Hz
    enable_preprocessing: bool = False
    model: SarvamTTSModels | str = "bulbul:v2"  # Default to v2 as it has more recent speakers
    api_key: str | None = None
    base_url: str = SARVAM_TTS_BASE_URL


class TTS(tts.TTS):
    """Sarvam.ai Text-to-Speech implementation.

    This class provides text-to-speech functionality using the Sarvam.ai API.
    Sarvam.ai specializes in high-quality TTS for Indian languages.

    Args:
        target_language_code: BCP-47 language code, e.g., "hi-IN"
        model: Sarvam TTS model to use
        speaker: Voice to use for synthesis
        speech_sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels (Sarvam outputs mono)
        pitch: Voice pitch adjustment (-20.0 to 20.0)
        pace: Speech rate multiplier (0.5 to 2.0)
        loudness: Volume multiplier (0.5 to 2.0)
        enable_preprocessing: Whether to use text preprocessing
        api_key: Sarvam.ai API key (falls back to SARVAM_API_KEY env var)
        base_url: API endpoint URL
        http_session: Optional aiohttp session to use
    """

    def __init__(
        self,
        *,
        target_language_code: str,
        model: SarvamTTSModels | str = "bulbul:v2",
        speaker: SarvamTTSSpeakers | str = "manisha",
        speech_sample_rate: int = 22050,
        num_channels: int = 1,  # Sarvam output is mono WAV
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
        enable_preprocessing: bool = False,
        api_key: str | None = None,
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
            raise ValueError(
                f"Speaker '{speaker}' is not compatible with model '{model}'. "
                f"Please choose a compatible speaker from: {', '.join(MODEL_SPEAKER_COMPATIBILITY.get(model, {}).get('all', []))}"
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
    async def synthesize(self, text: str, *, conn_options: Optional[APIConnectOptions] = None) -> AsyncIterator[tts.SynthesizedAudio]:
        """Synthesize text to audio using Sarvam.ai TTS API.

        Args:
            text: The text to synthesize
            conn_options: Connection options for the API request (for interface compatibility)

        Returns:
            An async iterator yielding audio frames

        Raises:
            APIConnectionError: On network connection errors
            APIStatusError: On API errors (non-200 status)
            APITimeoutError: On API timeout
        """
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        async for audio_event in self._synthesize_impl(text, conn_options=conn_options):
            yield audio_event

    async def _synthesize_impl(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> AsyncIterable[tts.SynthesizedAudio]:
        payload = {
            "target_language_code": self._opts.target_language_code,
            "text": text,
            "speaker": self._opts.speaker,
            "pitch": self._opts.pitch,
            "pace": self._opts.pace,
            "loudness": self._opts.loudness,
            "speech_sample_rate": self._opts.speech_sample_rate,
            "enable_preprocessing": self._opts.enable_preprocessing,
            "model": self._opts.model,
        }

        headers = {
            "api-subscription-key": self._opts.api_key,
            "Content-Type": "application/json",
        }

        _request_id = ""  # Variable to store request_id

        try:
            async with self._ensure_session().post(
                url=self._opts.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    self._logger.error(f"Sarvam TTS API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Sarvam TTS API Error: {error_text}", status_code=res.status
                    )

                response_json = await res.json()
                _request_id = response_json.get("request_id", "")  # Store request_id

                # Sarvam returns a list of base64 audios, we'll take the first one.
                if (
                    not response_json.get("audios")
                    or not isinstance(response_json["audios"], list)
                    or len(response_json["audios"]) == 0
                ):
                    self._logger.error("Sarvam TTS API response missing or invalid 'audios' field")
                    raise APIConnectionError("Sarvam TTS API response invalid: no audio data")

                base64_wav = response_json["audios"][0]
                wav_bytes = base64.b64decode(base64_wav)

                # Parse WAV and generate AudioFrames
                # Standard frame duration for WebRTC is 20ms, but can be 10ms. Let's use 20ms.
                frame_duration_ms = 20

                with io.BytesIO(wav_bytes) as wav_io:
                    with wave.open(wav_io, "rb") as wf:
                        sample_rate = wf.getframerate()
                        num_channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()  # Bytes per sample

                        if sample_rate != self._opts.speech_sample_rate:
                            self._logger.warning(
                                f"Sarvam TTS output sample rate {sample_rate} differs "
                                f"from requested {self._opts.speech_sample_rate}"
                            )
                            # Use actual sample rate from WAV for frame calculation

                        samples_per_channel_val = (
                            sample_rate * frame_duration_ms
                        ) // 1000  # Renamed for clarity
                        # For mono, samples_per_channel_val is samples_per_frame.
                        # If stereo, it would be samples_per_frame / num_channels.
                        # Since num_channels is 1, samples_per_channel_val is correct here.

                        bytes_per_frame = samples_per_channel_val * num_channels * sample_width

                        while True:
                            frame_data = wf.readframes(
                                samples_per_channel_val
                            )  # Read based on samples per channel
                            if not frame_data:
                                break

                            current_length = len(frame_data)
                            if current_length < bytes_per_frame:
                                # Pad with silence (zeros)
                                padding_needed = bytes_per_frame - current_length
                                frame_data += b"\x00" * padding_needed

                            audio_frame = rtc.AudioFrame(
                                data=frame_data,
                                sample_rate=sample_rate,  # Use actual sample rate from WAV
                                num_channels=num_channels,
                                samples_per_channel=samples_per_channel_val,
                            )
                            yield tts.SynthesizedAudio(request_id=_request_id, frame=audio_frame)

        except asyncio.TimeoutError as e:
            self._logger.error(f"Sarvam TTS API timeout: {e}")
            raise APITimeoutError("Sarvam TTS API request timed out") from e
        except aiohttp.ClientError as e:
            self._logger.error(f"Sarvam TTS API client error: {e}")
            raise APIConnectionError(f"Sarvam TTS API connection error: {e}") from e
        except Exception as e:
            self._logger.error(f"Error during Sarvam TTS synthesis: {e}")
            raise APIConnectionError(f"Unexpected error in Sarvam TTS: {e}") from e
