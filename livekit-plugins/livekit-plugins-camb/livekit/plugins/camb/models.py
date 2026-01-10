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
from typing import Literal

# Speech models supported by Camb.ai MARS series
SpeechModel = Literal[
    "mars-flash",  # Faster inference
    "mars-pro",  # Higher quality
    "mars-instruct",  # Supports user_instructions
]

# Audio output formats
OutputFormat = Literal[
    "pcm_s16le",  # 16-bit PCM (recommended for LiveKit)
    "pcm_s32le",  # 32-bit PCM (highest quality)
    "wav",  # WAV with headers
    "flac",  # Lossless compression
    "adts",  # Streaming format
]

# Voice ID type (integers in Camb.ai API)
CambVoiceId = int


@dataclass
class VoiceInfo:
    """Voice metadata from Camb.ai API."""

    id: int
    """Unique voice identifier (integer)."""
    name: str
    """Human-readable voice name."""
    gender: str | None = None
    """Voice gender category (mapped from: 0=Not Specified, 1=Male, 2=Female, 9=Not Applicable)."""
    language: int | None = None
    """Language code (integer returned by API)."""


@dataclass
class _TTSOptions:
    """Internal TTS configuration options."""

    voice_id: int
    language: str
    speech_model: str
    speed: float
    output_format: str
    user_instructions: str | None
    enhance_named_entities: bool


# Constants
DEFAULT_VOICE_ID = 147320
DEFAULT_LANGUAGE = "en-us"
DEFAULT_MODEL: SpeechModel = "mars-flash"  # Faster inference
DEFAULT_OUTPUT_FORMAT: OutputFormat = "pcm_s16le"
SAMPLE_RATE = 24000  # 24kHz - standard for Camb.ai PCM output
NUM_CHANNELS = 1
