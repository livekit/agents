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

"""Meta Muse Voice Transcribe plugin for LiveKit Agents."""

from .stt import STT, SpeechStream
from .version import __version__

__all__ = ["STT", "SpeechStream", "__version__"]

from livekit.agents import Plugin

from .log import logger


class MetaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(MetaPlugin())

# Cleanup docs of unexported modules.
_module = dir()
NOT_IN_ALL = [name for name in _module if name not in __all__]

__pdoc__ = {}
for name in NOT_IN_ALL:
    __pdoc__[name] = False
