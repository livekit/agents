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

"""Google AI plugin for LiveKit Agents

Supports Gemini, Cloud Speech-to-Text, and Cloud Text-to-Speech.

See https://docs.livekit.io/agents/integrations/stt/google/ for more information.
"""

from . import beta
from .llm import LLM
from .stt import STT, SpeechStream
from .tools import _LLMTool
from .tts import TTS
from .version import __version__

__all__ = ["STT", "TTS", "SpeechStream", "__version__", "beta", "LLM", "_LLMTool"]
from livekit.agents import Plugin

from .log import logger


class GooglePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(GooglePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
