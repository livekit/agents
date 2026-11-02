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

# Chat-capable vision-language models on the VLM Run gateway
# (GET https://gateway.vlm.run/v1/openai/models, task == "chat").
ChatModels = Literal[
    "qwen/qwen3.8-27b",
    "qwen/qwen3.5-0.8b",
    "google/gemma-4-26b-a4b-it",
    "google/diffusiongemma-26b-a4b-it",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.7-flash",
    "google/gemini-3.8-flash",
    "google/gemini-4-argon",
    "google/gemini-4-carbon",
    "google/gemini-robotics-er-2-preview",
    "moonshotai/kimi-k3",
    "minimax/minimax-m3",
    "meta/muse-glimmer-30b",
    "meta/muse-spark-1.2",
]

# Sampling fields each model accepts, from its `supported_parameters`. The gateway answers a
# sampling field the model doesn't accept with `400 capability_violation`, so the LLM refuses
# it at construction instead of failing on the first turn. Unlisted models are not checked.
_TEMPERATURE_TOP_P = frozenset({"temperature", "top_p"})

SUPPORTED_SAMPLING: dict[str, frozenset[str]] = {
    "qwen/qwen3.8-27b": _TEMPERATURE_TOP_P,
    "qwen/qwen3.5-0.8b": _TEMPERATURE_TOP_P,
    "google/gemma-4-26b-a4b-it": _TEMPERATURE_TOP_P,
    "google/diffusiongemma-26b-a4b-it": frozenset({"top_p"}),
    "google/gemini-3.5-flash-lite": _TEMPERATURE_TOP_P,
    "google/gemini-3.7-flash": _TEMPERATURE_TOP_P,
    "google/gemini-3.8-flash": _TEMPERATURE_TOP_P,
    "google/gemini-4-argon": _TEMPERATURE_TOP_P,
    "google/gemini-4-carbon": _TEMPERATURE_TOP_P,
    "google/gemini-robotics-er-2-preview": _TEMPERATURE_TOP_P,
    "moonshotai/kimi-k3": _TEMPERATURE_TOP_P,
    "minimax/minimax-m3": _TEMPERATURE_TOP_P,
    "meta/muse-glimmer-30b": _TEMPERATURE_TOP_P,
    "meta/muse-spark-1.2": _TEMPERATURE_TOP_P,
}
