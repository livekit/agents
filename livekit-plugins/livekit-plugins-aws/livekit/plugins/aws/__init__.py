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

"""AWS plugin for LiveKit Agents

Support for AWS AI including Bedrock, Polly, Transcribe and optionally Nova Sonic.

See https://docs.livekit.io/agents/integrations/aws/ for more information.
"""

import typing  # noqa: I001


if typing.TYPE_CHECKING:
    from .experimental import realtime


def __getattr__(name: str) -> typing.Any:
    if name == "realtime":
        try:
            from .experimental import realtime
        except ImportError as e:
            raise ImportError(
                "The 'realtime' module requires optional dependencies. "
                "Please install them with: pip install 'livekit-plugins-aws[realtime]'"
            ) from e

        return realtime

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .llm import LLM  # noqa: E402
from .stt import STT, SpeechStream  # noqa: E402
from .tts import TTS, ChunkedStream  # noqa: E402
from .version import __version__  # noqa: E402

__all__ = ["STT", "SpeechStream", "TTS", "ChunkedStream", "LLM", "realtime", "__version__"]

from livekit.agents import Plugin  # noqa: E402

from .log import logger  # noqa: E402


class AWSPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AWSPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
