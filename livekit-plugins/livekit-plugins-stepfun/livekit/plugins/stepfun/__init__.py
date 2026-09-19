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

"""StepFun plugin for LiveKit Agents framework."""

from livekit.agents import Plugin

from . import realtime, tools
from .log import logger
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    DOMESTIC_BASE_URL,
    OVERSEAS_BASE_URL,
    StepAudioRealtimeModels,
    StepAudioRealtimeVoices,
)
from .version import __version__

__all__ = [
    "realtime",
    "tools",
    "StepAudioRealtimeModels",
    "StepAudioRealtimeVoices",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "DEFAULT_BASE_URL",
    "DOMESTIC_BASE_URL",
    "OVERSEAS_BASE_URL",
    "__version__",
]


class StepFunPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(StepFunPlugin())
