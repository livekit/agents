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

"""Sarvam.ai plugin for LiveKit Agents.

Support for speech-to-text, text-to-speech, and LLM with Sarvam.ai.

Sarvam.ai provides high-quality STT and TTS for Indian languages and
OpenAI-compatible LLMs.

For API access, visit https://sarvam.ai/
"""

from livekit.agents import Plugin

from .llm import LLM, SarvamLLMModels
from .log import logger
from .stt import STT
from .stt_streaming import RealtimeSpeechStream, STTRealtime
from .tts import TTS
from .version import __version__

# Deprecated compatibility aliases. Prefer `STTRealtime` and
# `RealtimeSpeechStream`.
STTStreaming = STTRealtime
StreamingSpeechStream = RealtimeSpeechStream

__all__ = [
    "STT",
    "STTRealtime",
    "RealtimeSpeechStream",
    "STTStreaming",
    "StreamingSpeechStream",
    "TTS",
    "LLM",
    "SarvamLLMModels",
    "__version__",
]


class SarvamPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(SarvamPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
