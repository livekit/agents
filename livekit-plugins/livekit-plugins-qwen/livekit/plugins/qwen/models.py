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

"""Model ids, regions and defaults for Qwen on Alibaba Cloud Model Studio."""

from typing import Literal

QwenRegion = Literal["intl", "cn"]
"""Model Studio deployment region.

``intl`` is Singapore (keys created on alibabacloud.com), ``cn`` is Beijing (keys created on
aliyun.com). API keys are region-bound: a key from one region is rejected by the other.
"""

DEFAULT_REGION: QwenRegion = "intl"

REALTIME_BASE_URLS: dict[str, str] = {
    "intl": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
    "cn": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
}
"""Public realtime WebSocket endpoints per region. Workspace-dedicated domains
(``wss://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime``) go in ``base_url``."""

COMPAT_BASE_URLS: dict[str, str] = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
"""OpenAI-compatible chat-completions endpoints per region (LLM only)."""

STTModels = Literal["qwen3-asr-flash-realtime"]
DEFAULT_STT_MODEL: STTModels = "qwen3-asr-flash-realtime"
STT_SAMPLE_RATE = 16000
"""The realtime ASR model accepts 16000 or 8000 Hz only; 16000 is used and LiveKit resamples."""

TTSModels = Literal["qwen3-tts-flash-realtime"]
DEFAULT_TTS_MODEL: TTSModels = "qwen3-tts-flash-realtime"
TTS_SAMPLE_RATE = 24000
"""Model Studio's default and the widest-supported output rate for the Qwen3 TTS series."""

TTSVoices = Literal["Cherry", "Serena", "Ethan", "Chelsie", "Kiki", "Jennifer", "Rocky", "Ryan"]
"""Built-in voices. Cherry, Serena, Ethan and Chelsie cover both Mandarin and English; see
https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-list for the full list."""
DEFAULT_TTS_VOICE: TTSVoices = "Cherry"

TTSLanguageTypes = Literal[
    "Auto",
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
]
"""Model Studio's ``language_type`` values. ``Auto`` detects per request; naming the language
improves quality on single-language text."""
DEFAULT_TTS_LANGUAGE_TYPE: TTSLanguageTypes = "Auto"

LLMModels = Literal["qwen3-max", "qwen-plus", "qwen-flash", "qwen-turbo"]
DEFAULT_LLM_MODEL: LLMModels = "qwen-plus"
