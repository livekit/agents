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

from typing import Literal

TTSModels = Literal[
    "KittenML/kitten-tts-nano-0.1",
    "KittenML/kitten-tts-nano-0.2",
]

TTSVoices = Literal[
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]

HG_MODEL = "KittenML/kitten-tts-nano-0.2"

ONNX_FILENAME = "kitten_tts_nano_v0_2.onnx"
VOICES_FILENAME = "voices.npz"


def resolve_model_name(model_name: str) -> str:
    legacy_models = {
        "nano-0.1": "KittenML/kitten-tts-nano-0.1",
        "nano-0.2": "KittenML/kitten-tts-nano-0.2",
    }

    if model_name in legacy_models:
        return legacy_models[model_name]

    if "/" not in model_name:
        return f"KittenML/{model_name}"

    return model_name
