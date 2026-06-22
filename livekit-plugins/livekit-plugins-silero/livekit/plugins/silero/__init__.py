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

"""Silero VAD plugin for LiveKit Agents

See https://docs.livekit.io/agents/build/turns/vad/ for more information.
"""

import warnings

from .vad import VAD, VADStream
from .version import __version__

__all__ = ["VAD", "VADStream", "__version__"]

from livekit.agents import Plugin

from .log import logger

warnings.warn(
    "livekit-plugins-silero is deprecated and will be removed in v2.0. "
    "AgentSession now defaults to the bundled silero VAD, so you can drop the "
    "explicit `vad=` argument entirely; pass `vad=None` to opt out, or use "
    '`from livekit.agents import inference; inference.VAD(model="silero", ...)`'
    " to customise options.",
    DeprecationWarning,
    stacklevel=2,
)


class SileroPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(SileroPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
