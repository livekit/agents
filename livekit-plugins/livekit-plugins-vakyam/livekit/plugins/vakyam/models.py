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

from typing import Literal

TTSModels = Literal["raaga-v1"]
TTSLanguages = Literal["bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "mr-IN", "ta-IN", "te-IN"]
TTSSampleRates = Literal[8000, 16000, 24000, 48000]

DEFAULT_MODEL: TTSModels = "raaga-v1"
DEFAULT_VOICE = "Archana"
DEFAULT_LANGUAGE: TTSLanguages = "ta-IN"
DEFAULT_SAMPLE_RATE: TTSSampleRates = 24000
DEFAULT_SPEED = 1.0
DEFAULT_BASE_URL = "https://api.vakyam.ai"

TTS_STREAM_PATH = "/v1/tts/stream"
TTS_WEBSOCKET_PATH = "/v1/tts/websocket"
MAX_TEXT_CHARACTERS = 3000
CUSTOM_VOICE_PREFIX = "vc_"
MIN_SPEED = 0.5
MAX_SPEED = 2.0

# Application-level keepalive. The server closes idle sockets after ~60s.
KEEPALIVE_INTERVAL_SECONDS = 20.0

SUPPORTED_MODELS: frozenset[str] = frozenset({"raaga-v1"})
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "mr-IN", "ta-IN", "te-IN"}
)
SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({8000, 16000, 24000, 48000})
