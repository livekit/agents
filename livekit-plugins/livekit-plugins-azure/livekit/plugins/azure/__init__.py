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

"""Azure plugin for LiveKit Agents

Support for Azure AI including Azure Speech and optionally the Azure Voice Live Realtime API. For Azure OpenAI, see the [OpenAI plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai).

See https://docs.livekit.io/agents/integrations/azure/ for more information.
"""

import importlib
import typing

from . import responses
from .stt import STT, SpeechStream
from .tts import TTS
from .version import __version__

if typing.TYPE_CHECKING:
    from . import realtime

__all__ = ["STT", "SpeechStream", "TTS", "realtime", "responses", "__version__"]

from livekit.agents import Plugin

from .log import logger


def __getattr__(name: str) -> typing.Any:
    # the realtime module needs the optional `realtime` extra, so import it on first access
    if name == "realtime":
        try:
            return importlib.import_module(f"{__name__}.realtime")
        except ImportError as e:
            raise ImportError(
                "The 'realtime' module requires optional dependencies. "
                "Please install them with: pip install 'livekit-plugins-azure[realtime]'"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class AzurePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AzurePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
