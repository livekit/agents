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

"""Avaluma plugin for LiveKit Agents

Support for the Avaluma virtual avatar.

See https://docs.livekit.io/agents/models/avatar/plugins/avaluma/ for more information.
"""

from .avatar import AvalumaException, AvatarSession
from .version import __version__

__all__ = [
    "AvalumaException",
    "AvatarSession",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class AvalumaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AvalumaPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
