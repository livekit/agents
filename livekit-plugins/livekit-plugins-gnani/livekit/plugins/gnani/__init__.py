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

"""Gnani plugin for LiveKit Agents

Support for speech-to-text and text-to-speech with [Gnani](https://gnani.ai/).

Gnani provides high-accuracy STT and low-latency TTS for Indian languages,
including multilingual and code-switching scenarios.

See https://docs.livekit.io/agents/integrations/stt/gnani/ for more information.
"""

from .stt import (
    STREAM_SUPPORTED_LANGUAGES,
    STT,
    SUPPORTED_LANGUAGES,
    GnaniSTTFormat,
    GnaniSTTLanguages,
    SpeechStream,
)
from .tts import (
    DEFAULT_MODEL,
    SUPPORTED_TTS_LANGUAGES,
    TIMBRE_V20_VOICES,
    TIMBRE_V25_VOICES,
    TTS,
    GnaniTTSBitrates,
    GnaniTTSContainers,
    GnaniTTSEncodings,
    GnaniTTSSynthesizeMethod,
    GnaniTTSVoices,
    SynthesizeStream,
)
from .version import __version__

__all__ = [
    "DEFAULT_MODEL",
    "GnaniSTTFormat",
    "GnaniSTTLanguages",
    "GnaniTTSBitrates",
    "GnaniTTSContainers",
    "GnaniTTSEncodings",
    "GnaniTTSSynthesizeMethod",
    "GnaniTTSVoices",
    "STT",
    "STREAM_SUPPORTED_LANGUAGES",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_TTS_LANGUAGES",
    "TIMBRE_V20_VOICES",
    "TIMBRE_V25_VOICES",
    "TTS",
    "SpeechStream",
    "SynthesizeStream",
    "__version__",
]


from livekit.agents import Plugin

from .log import logger


class GnaniPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(GnaniPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
