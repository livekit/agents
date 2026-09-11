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

"""Qwen plugin for LiveKit Agents: realtime STT, TTS and LLM on Alibaba Cloud Model Studio.

Importing this package registers the plugin via ``Plugin.register_plugin``.

Set ``DASHSCOPE_API_KEY`` (or pass ``api_key``). Keys are region-bound: ``region="intl"``
(default, Singapore) for keys from alibabacloud.com, ``region="cn"`` (Beijing) for keys from
aliyun.com. Pass ``base_url`` to use a workspace-dedicated domain.
"""

from .llm import LLM
from .models import (
    DEFAULT_LLM_MODEL,
    DEFAULT_REGION,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_LANGUAGE_TYPE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    LLMModels,
    QwenRegion,
    STTModels,
    TTSLanguageTypes,
    TTSModels,
    TTSVoices,
)
from .stt import STT, SpeechStream
from .tts import TTS, SynthesizeStream
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "TTS",
    "SynthesizeStream",
    "LLM",
    "QwenRegion",
    "STTModels",
    "TTSModels",
    "TTSVoices",
    "TTSLanguageTypes",
    "LLMModels",
    "DEFAULT_REGION",
    "DEFAULT_STT_MODEL",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TTS_VOICE",
    "DEFAULT_TTS_LANGUAGE_TYPE",
    "DEFAULT_LLM_MODEL",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class QwenPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(QwenPlugin())

# Hide non-exported names from the generated pdoc output.
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
