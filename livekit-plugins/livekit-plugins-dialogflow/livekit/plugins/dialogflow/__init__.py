# Copyright 2024 LiveKit, Inc.
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

"""Google Dialogflow CX plugin for LiveKit Agents.

Integrates Dialogflow CX as an LLM provider in an STT -> LLM -> TTS pipeline.
Dialogflow CX is an intent-based conversational AI engine — all conversation logic
is managed in the Dialogflow CX console, not via the ``instructions`` field.

See https://cloud.google.com/dialogflow/cx/docs for Dialogflow CX documentation.
"""

from .llm import LLM, DialogflowLLMStream as LLMStream
from .version import __version__

__all__ = [
    "LLM",
    "LLMStream",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class DialogflowPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DialogflowPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__: dict[str, bool] = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
