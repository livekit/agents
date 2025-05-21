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

"""Whisper plugin for LiveKit Agents

Provides STT functionality using OpenAI's Whisper models.
The models are run locally, no API key is required.
"""

from .stt import STT
from .version import __version__

__all__ = [
    "STT",
    "__version__",
]

from livekit.agents import Plugin
import logging # Using standard logging for now

logger = logging.getLogger(__name__)


class WhisperPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            name="whisper",
            version=__version__,
            pkg_path="livekit.plugins.whisper", # Follows pattern of other plugins
            logger=logger,
        )

    def download_files(self):
        # This method could be used to trigger model downloads if needed,
        # but whisper.load_model() handles downloads automatically.
        # For now, we can leave it empty or add a log message.
        logger.info("Whisper models are downloaded on-demand by the whisper library.")
        pass


Plugin.register_plugin(WhisperPlugin())

# Cleanup docs of unexported modules
# This helps in generating cleaner API documentation
_module_dir = dir()
NOT_IN_ALL = [m for m in _module_dir if not m.startswith("_") and m not in __all__]

__pdoc__ = {}
for n in NOT_IN_ALL:
    __pdoc__[n] = False
