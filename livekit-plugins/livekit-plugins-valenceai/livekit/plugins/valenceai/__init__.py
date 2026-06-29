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

"""ValenceAI plugin for LiveKit Agents.

This plugin provides real-time emotion detection for audio using ValenceAI's
streaming WebSocket API. It wraps an underlying STT provider (e.g., Deepgram)
and enriches transcriptions with emotion tags.

Example:
    from livekit.plugins import valenceai, deepgram

    emotion_stt = valenceai.STT(
        underlying_stt=deepgram.STT(),
        api_key="your-valence-api-key",  # or set VALENCE_API_KEY env var
    )

    session = AgentSession(stt=emotion_stt, ...)
"""

from .client import EmotionModel, ValenceWebSocketClient
from .stt import STT, EmotionAwareRecognizeStream
from .version import __version__

__all__ = [
    "STT",
    "EmotionAwareRecognizeStream",
    "ValenceWebSocketClient",
    "EmotionModel",
    "__version__",
]


from livekit.agents import Plugin

from .log import logger


class ValenceAIPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ValenceAIPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
