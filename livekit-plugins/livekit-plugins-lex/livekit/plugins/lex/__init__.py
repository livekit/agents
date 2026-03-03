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

"""Amazon Lex V2 plugin for LiveKit Agents.

Integrates Amazon Lex V2 as an LLM provider in an STT -> LLM -> TTS pipeline.
Lex V2 is an intent-based conversational AI engine — all conversation logic
is managed in the AWS Lex V2 console, not via the ``instructions`` field.

See https://docs.aws.amazon.com/lex/ for Amazon Lex V2 documentation.
"""

from .llm import LLM, LexLLMStream as LLMStream
from .version import __version__

__all__ = [
    "LLM",
    "LLMStream",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class LexPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(LexPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__: dict[str, bool] = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
