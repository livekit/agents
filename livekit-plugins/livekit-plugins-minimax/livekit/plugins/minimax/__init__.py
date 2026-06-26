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

"""Minimax plugin for LiveKit Agents

See [Plugin Docs URL - when available] for more information.
"""

# 1. Import the core TTS class from our tts.py module.
# 2. Import the version number.
from .tts import TTS
from .version import __version__

# 3. Define the public API of the package.
#    Since we only implement TTS, we only expose the TTS class and the version.
__all__ = ["TTS", "__version__"]

from livekit.agents import Plugin

# 4. Import the package-specific logger.
from .log import logger


# 5. Define and register the plugin.
class MiniMaxPlugin(Plugin):
    def __init__(self) -> None:
        # The super() call requires the plugin name, version, package path, and logger.
        super().__init__(__name__, __version__, __package__, logger)


# Register an instance of our plugin with the LiveKit Agents framework.
Plugin.register_plugin(MiniMaxPlugin())

# 6. (Optional but recommended) pdoc configuration to hide internal modules
#    from the generated documentation. This is standard boilerplate.
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
