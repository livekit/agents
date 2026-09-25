# Copyright 2026 LiveKit, Inc.
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

"""Airy text-to-speech plugin for LiveKit Agents."""

from livekit.agents import Plugin

from .log import logger
from .tts import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_STYLE,
    DEFAULT_VOICE,
    TTS,
    ChunkedStream,
    Language,
    Style,
)
from .version import __version__

__all__ = [
    "TTS",
    "ChunkedStream",
    "Language",
    "Style",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "DEFAULT_STYLE",
    "__version__",
]


class AiryPlugin(Plugin):
    """Register the Airy provider with LiveKit Agents."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AiryPlugin())

_module = dir()
NOT_IN_ALL = [name for name in _module if name not in __all__]

__pdoc__ = dict.fromkeys(NOT_IN_ALL, False)
