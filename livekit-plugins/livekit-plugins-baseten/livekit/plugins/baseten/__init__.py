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

from .llm import LLM
from .log import logger
from .models import LLMModels
from .stt import STT, SpeechStream
from .tts import TTS
from .version import __version__

__all__ = [
    "LLM",
    "STT",
    "SpeechStream",
    "logger",
    "TTS",
    "LLMModels",
    "__version__",
]

from livekit.agents import Plugin


class BasetenPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)


Plugin.register_plugin(BasetenPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
