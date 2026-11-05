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
    "ragini-v1",  # 24 kHz
    "ragini-lite",  # 22.05 kHz
]

# Native output rate per TTS model. Ragini resamples on request, but asking for
# the model's own rate avoids a resampling pass on the server.
TTS_DEFAULT_SAMPLE_RATES: dict[str, int] = {
    "ragini-v1": 24000,
    "ragini-lite": 22050,
}

TTS_FALLBACK_SAMPLE_RATE = 24000

# Language hints accepted by Akshara's `lang_tags`. The server rejects anything
# outside this set, so `STT` only forwards a tag that appears here.
STT_LANGUAGES = frozenset({"bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"})

STTLanguages = Literal["bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"]

TTSLanguages = Literal["bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"]

TTS_LANGUAGES = frozenset({"bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"})

TTSVoices = Literal[
    "roohi",
    "amaya",
    "kiyansh",
    "neeraj",
    "manya",
    "nidhi",
    "ira",
    "trisha",
    "charvi",
]
