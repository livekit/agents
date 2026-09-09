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

"""Palabra plugin for LiveKit Agents: realtime STT and TTS on the ``palabra-ai`` SDK.

Importing this package registers the plugin via ``Plugin.register_plugin``.
Docs: https://docs.livekit.io/agents/models/stt/ and .../tts/.
"""

from .models import (
    DEFAULT_DEACCENT_STRENGTH,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    TTSLanguages,
    TTSModels,
)
from .stt import STT, SpeechStream
from .tts import TTS, SynthesizeStream
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "TTS",
    "SynthesizeStream",
    "TTSModels",
    "TTSLanguages",
    "DEFAULT_LANGUAGE",
    "DEFAULT_VOICE_ID",
    "DEFAULT_MODEL",
    "DEFAULT_DEACCENT_STRENGTH",
    "DEFAULT_SAMPLE_RATE",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class PalabraPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(PalabraPlugin())

# Hide non-exported names from the generated pdoc output.
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
