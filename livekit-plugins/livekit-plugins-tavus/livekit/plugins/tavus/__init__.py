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

"""Tavus virtual avatar plugin for LiveKit Agents

See https://docs.livekit.io/agents/integrations/avatar/tavus/ for more information.
"""

from .avatar import AvatarSession, TavusException
from .version import __version__

__all__ = [
    "TavusException",
    "AvatarSession",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class TavusPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(TavusPlugin())
