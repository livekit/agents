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

"""FunASR plugin for LiveKit Agents.

Local, fully-offline multilingual speech-to-text using FunASR models such as
SenseVoice (Chinese, Cantonese, English, Japanese, Korean and more).
See https://github.com/modelscope/FunASR for more information.
"""

from .stt import _DEFAULT_MODEL, FunASRSTT, FunASRSTT as STT, _load_model
from .version import __version__

__all__ = ["FunASRSTT", "STT", "__version__"]

from livekit.agents import Plugin

from .log import logger


class FunASRPlugin(Plugin):
    """Register the FunASR integration and its model-download hook."""

    def __init__(self) -> None:
        """Create the LiveKit plugin registration."""
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        """Download the default SenseVoice model for offline agent startup."""
        _load_model(_DEFAULT_MODEL, "cpu")


Plugin.register_plugin(FunASRPlugin())
