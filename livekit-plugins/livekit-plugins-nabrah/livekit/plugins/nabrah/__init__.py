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

"""Nabrah STT: original recognizers and conversational STT over the backend WebSocket."""

from livekit.agents import Plugin

from ._factory import STT, NabrahRecognitionModel
from .backchannel import BackchannelController
from .log import logger
from .stt import SpeechStream
from .stt_conv import SpeechStreamConv, STTConv
from .version import __version__


class NabrahPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(NabrahPlugin())

__all__ = [
    "STT",
    "NabrahRecognitionModel",
    "SpeechStream",
    "STTConv",
    "SpeechStreamConv",
    "BackchannelController",
    "__version__",
]
