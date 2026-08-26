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

"""Floe plugin for LiveKit Agents

Route LiveKit's LLM, STT, and TTS through Floe for metered spend and
budget-guarded inference, either keyless (Floe holds the provider keys) or BYOK
(bring your own provider key). Includes a usage reconciler that reconciles
LiveKit-reported token usage against Floe pricing.

See https://docs.livekit.io for more information.
"""

from livekit.agents import Plugin

from .log import logger
from .metering import FloeUsageReconciler
from .receipt import enable_cost_receipts
from .services import LLM
from .stt import STT
from .tts import TTS
from .version import __version__

__all__ = [
    "LLM",
    "STT",
    "TTS",
    "FloeUsageReconciler",
    "enable_cost_receipts",
    "__version__",
]


class FloePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(FloePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
