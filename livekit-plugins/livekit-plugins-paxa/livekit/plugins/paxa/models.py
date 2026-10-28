# Copyright 2026 LiveKit, Inc.
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

from typing import Literal

TTSModels = Literal["paxa-tts-flash-v2", "paxa-tts-flash-v1"]
TTSLanguages = Literal["auto", "th", "en", "zh"]

DEFAULT_MODEL: TTSModels = "paxa-tts-flash-v2"
DEFAULT_VOICE = "foithong"
DEFAULT_LANGUAGE: TTSLanguages = "auto"
DEFAULT_SPEED = 1.0
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_BASE_URL = "https://api.paxalabs.com"

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"auto", "th", "en", "zh"})
MIN_SPEED = 0.5
MAX_SPEED = 1.5

# Paxa serves pcm as 16-bit little-endian mono, at these rates in Hz.
SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})
MODEL_SAMPLE_RATES: dict[str, frozenset[int]] = {
    "paxa-tts-flash-v2": SUPPORTED_SAMPLE_RATES,
    "paxa-tts-flash-v1": frozenset({24000}),
}
NUM_CHANNELS = 1
