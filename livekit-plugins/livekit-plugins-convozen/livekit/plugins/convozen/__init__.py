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

"""ConvoZen plugin for LiveKit Agents

Support for speech-to-text and text-to-speech with
[ConvoZen's voice platform](https://convozen.ai/): **Akshara** for recognition and
**Ragini** for synthesis, covering nine Indian languages plus code-mixed speech.

Both models are batch (whole-utterance) rather than incremental, so a VAD is
required on the `AgentSession` — LiveKit then wraps the STT in a `StreamAdapter`
automatically:

```python
session = AgentSession(
    stt=convozen.STT(language="hi"),
    llm=...,
    tts=convozen.TTS(voice="roohi", language="hi"),
    vad=silero.VAD.load(),
)
```

See https://docs.livekit.io/agents/models/ for more information.
"""

from .models import (
    STTLanguages,
    STTModels,
    TTSLanguages,
    TTSModels,
    TTSVoices,
)
from .stt import STT
from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "ChunkedStream",
    "STTModels",
    "STTLanguages",
    "TTSModels",
    "TTSLanguages",
    "TTSVoices",
    "__version__",
]


from livekit.agents import Plugin

from .log import logger


class ConvozenPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ConvozenPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
