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

from dataclasses import dataclass
from typing import Literal

# Typecast TTS models
TTSModels = Literal["ssfm-v21"]

# Audio format options
AudioFormat = Literal["wav", "mp3"]

# Supported languages (ISO 639-3 codes)
TTSLanguages = Literal[
    "eng",  # English
    "kor",  # Korean
    "jpn",  # Japanese
    "zho",  # Chinese
    "spa",  # Spanish
    "deu",  # German
    "fra",  # French
    "ita",  # Italian
    "rus",  # Russian
    "ara",  # Arabic
    "por",  # Portuguese
    "nld",  # Dutch
    "pol",  # Polish
    "swe",  # Swedish
    "tur",  # Turkish
    "hin",  # Hindi
    "tha",  # Thai
    "vie",  # Vietnamese
    "ind",  # Indonesian
]


@dataclass
class PromptOptions:
    """
    Options for controlling the emotional expression in Typecast TTS synthesis.

    Attributes:
        emotion_preset: Emotion type (e.g., "normal", "happy", "sad", "angry")
        emotion_intensity: Intensity of the emotion (0.0 ~ 2.0, default: 1.0)
    """

    emotion_preset: str = "normal"
    emotion_intensity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "emotion_preset": self.emotion_preset,
            "emotion_intensity": self.emotion_intensity,
        }


@dataclass
class OutputOptions:
    """
    Options for controlling the audio output characteristics.

    Attributes:
        volume: Volume level (0 ~ 200, default: 100)
        audio_pitch: Pitch adjustment in semitones (-12 ~ +12, default: 0)
        audio_tempo: Speed multiplier (0.5x ~ 2.0x, default: 1.0)
        audio_format: Output format ("wav" or "mp3", default: "wav")
    """

    volume: int = 100
    audio_pitch: int = 0
    audio_tempo: float = 1.0
    audio_format: AudioFormat = "wav"

    def to_dict(self) -> dict:
        return {
            "volume": self.volume,
            "audio_pitch": self.audio_pitch,
            "audio_tempo": self.audio_tempo,
            "audio_format": self.audio_format,
        }
