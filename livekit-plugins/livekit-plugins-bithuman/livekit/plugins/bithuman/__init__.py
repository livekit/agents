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

"""BitHuman plugin for LiveKit Agents

See https://docs.livekit.io/agents/integrations/avatar/bithuman/ for more information.
"""

import sys

try:
    from .avatar import AvatarSession, BitHumanException
except ImportError:
    # the bithuman SDK publishes no build for Python 3.14+, so neither it nor the third-party
    # packages it pulls in are installed there
    if sys.version_info >= (3, 14):
        print(
            "the bithuman SDK is not available on Python "
            f"{sys.version_info.major}.{sys.version_info.minor}, "
            "livekit-plugins-bithuman requires Python 3.13 or older",
            flush=True,
        )
    raise

from .version import __version__

__all__ = [
    "BitHumanException",
    "AvatarSession",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class BitHumanPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(BitHumanPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
