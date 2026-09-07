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

"""SILMA AI plugin for LiveKit Agents

Arabic and English text-to-speech with SILMA TTS v2.

See https://silma.ai/ for more information.
"""

from .models import (
    ARABIC_VOICES,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    ENGLISH_VOICES,
    TTSArabicVoices,
    TTSEnglishVoices,
    TTSModels,
    TTSVoices,
)
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = [
    "TTS",
    "ChunkedStream",
    "SynthesizeStream",
    "TTSModels",
    "TTSVoices",
    "TTSArabicVoices",
    "TTSEnglishVoices",
    "ARABIC_VOICES",
    "ENGLISH_VOICES",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class SilmaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(SilmaPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
