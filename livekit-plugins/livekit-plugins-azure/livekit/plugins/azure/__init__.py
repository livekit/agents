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

from .stt import STT, SpeechStream
from .tts import TTS
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "TTS",
    "__version__",
]

from livekit.agents import Plugin

class AzurePlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__,__package__)

    def download_files(self):
        pass

Plugin.register_plugin(AzurePlugin())
