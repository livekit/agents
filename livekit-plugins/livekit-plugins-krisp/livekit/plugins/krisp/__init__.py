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

"""Krisp VIVA plugin for LiveKit Agents

This plugin provides real-time noise reduction
using Krisp's proprietary algorithms via the VIVA SDK.

Features:
    - KrispVivaFilterFrameProcessor: Real-time noise reduction FrameProcessor
"""

from livekit.agents import Plugin

from .krisp_instance import (
    KRISP_FRAME_DURATIONS,
    KRISP_SAMPLE_RATES,
    KrispSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)
from .log import logger
from .version import __version__
from .viva_filter import KrispVivaFilterFrameProcessor

__all__ = [
    "KrispVivaFilterFrameProcessor",
    "KrispSDKManager",
    "KRISP_SAMPLE_RATES",
    "KRISP_FRAME_DURATIONS",
    "int_to_krisp_sample_rate",
    "int_to_krisp_frame_duration",
    "__version__",
]

class KrispPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

Plugin.register_plugin(KrispPlugin())
