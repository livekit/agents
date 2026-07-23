# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Oracle Corporation and/or its affiliates.
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

"""
Oracle plug-ins for LiveKit Agents

Support for Oracle RTS, GenAI, and TTS services.
"""

from livekit.agents import Plugin

from .llm import LLM
from .log import logger
from .oracle_llm import BackEnd, Role
from .stt import STT
from .tts import TTS
from .utils import AuthenticationType
from .version import __version__

__all__ = ["STT", "LLM", "TTS", "AuthenticationType", "BackEnd", "Role", "__version__"]


class OraclePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(OraclePlugin())


# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]


__pdoc__ = {}
for n in NOT_IN_ALL:
    __pdoc__[n] = False
