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

from __future__ import annotations

from typing import Literal

from livekit.agents import __version__ as livekit_version

from .version import __version__

TTSModels = Literal[
    "silma-tts-v2-english",
    "silma-tts-v2-msa",
    "silma-tts-v2-ksa",
]
"""SILMA TTS v2 models.

- ``silma-tts-v2-english``: English.
- ``silma-tts-v2-msa``: Modern Standard Arabic.
- ``silma-tts-v2-ksa``: Saudi (Najdi) Arabic dialect.
"""

TTSEnglishVoices = Literal["james", "emma"]
"""Voices available on the English model."""

TTSArabicVoices = Literal[
    "sarah",
    "salma",
    "salwa",
    "saja",
    "sultan",
    "salman",
    "sulaiman",
    "salim",
]
"""Voices available on the Arabic (MSA and KSA) models."""

TTSVoices = Literal[TTSEnglishVoices, TTSArabicVoices]
"""Every pre-defined SILMA voice."""

ENGLISH_VOICES: frozenset[str] = frozenset({"james", "emma"})
ARABIC_VOICES: frozenset[str] = frozenset(
    {"sarah", "salma", "salwa", "saja", "sultan", "salman", "sulaiman", "salim"}
)

DEFAULT_BASE_URL = "https://api.silma.ai/tts/v2"
DEFAULT_MODEL: TTSModels = "silma-tts-v2-msa"
DEFAULT_VOICE: TTSVoices = "sarah"

# SILMA v2 renders a 24 kHz mono float32 waveform.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# `text` is capped by the API. Longer input is split before it is sent.
MAX_TEXT_CHARACTERS = 250

HTTP_STREAM_PATH = "/stream"
WEBSOCKET_PATH = "/ws/stream"

API_KEY_HEADER = "apiKey"

# Identifies plugin traffic to SILMA, and carries both versions so a report of
# "the plugin is misbehaving" can be tied to a specific plugin and framework
# build. Sent on the HTTP request and the WebSocket handshake alike.
USER_AGENT = f"LiveKit-Agents-SILMA/{__version__} livekit-agents/{livekit_version}"
