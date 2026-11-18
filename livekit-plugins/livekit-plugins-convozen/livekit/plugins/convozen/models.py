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

"""Model, language and voice identifiers for the ConvoZen platform."""

from __future__ import annotations

from typing import Literal

# The public ConvoZen developer API. Override with `base_url=` or the
# CONVOZEN_BASE_URL environment variable when pointing at another deployment.
DEFAULT_BASE_URL = "https://voice.convozen.ai/sdk-models/developer-api/api"

STTModels = Literal[
    "akshara-pro",
    "akshara",
]

TTSModels = Literal[
    "ragini-v2",  # 48 kHz
]

# Native output rate per TTS model, which is the rate always requested on the wire.
# Any other output rate is produced by resampling on the client.
TTS_DEFAULT_SAMPLE_RATES: dict[str, int] = {
    "ragini-v2": 48000,
}

TTS_FALLBACK_SAMPLE_RATE = 48000

# Language hints accepted by Akshara's `lang_tags`. The server rejects anything
# outside this set, so `STT` only forwards a tag that appears here.
STT_LANGUAGES = frozenset({"bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"})

STTLanguages = Literal["bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"]

# ragini-v2 is deployed for these languages only; anything else is a 400.
TTSLanguages = Literal["en", "hi", "kn", "ta", "te"]

TTS_LANGUAGES = frozenset({"en", "hi", "kn", "ta", "te"})

# ragini-v2 runs one checkpoint for en/hi and another for kn/ta/te, each with its
# own voice set. A voice from the other group is not rejected — the server treats
# it as a raw speaker tag and returns an arbitrary voice.
TTSVoices = Literal[
    # en, hi
    "shrajay",
    "siddharth",
    "parmi",
    "sherin",
    "bhavya",
    "shreya",
    # kn, ta, te (also: parmi, bhavya, shrajay, siddharth)
    "amaya",
    "roohi",
    "bhagya",
]

# Default voice per language, matching the server's own default for each group.
TTS_DEFAULT_VOICES: dict[str, str] = {
    "en": "shrajay",
    "hi": "shrajay",
    "kn": "amaya",
    "ta": "amaya",
    "te": "amaya",
}
