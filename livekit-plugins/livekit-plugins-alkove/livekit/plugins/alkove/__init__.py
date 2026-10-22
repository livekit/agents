# Copyright 2026 Eddy Bordi
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

"""Alkove plugin for LiveKit Agents

Focus keeps the user's voice and removes the other voices and the noise, as the
`noise_cancellation` of a room's audio input. Cue finds where they speak, as the session's
VAD. Both run on the agent's CPU, self-hosted, with no API key.

See https://alkove.ai for more information.
"""

from alkove import Models, SpeechParams, Timbre, Voiceprint
from livekit.agents import Plugin

from ._models import models
from .focus import Focus
from .log import logger
from .vad import VAD, VADStream
from .version import __version__

__all__ = [
    "Focus",
    "Models",
    "SpeechParams",
    "Timbre",
    "VAD",
    "VADStream",
    "Voiceprint",
    "models",
    "__version__",
]


class AlkovePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        import alkove

        alkove.fetch(["focus", "cue"])


Plugin.register_plugin(AlkovePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
