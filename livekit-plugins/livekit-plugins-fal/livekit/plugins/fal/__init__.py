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

from .sd_turbo_high_fps import SDTurboHighFPS, SDTurboHighFPSStream
from .version import __version__

__all__ = ["SDTurboHighFPS", "SDTurboHighFPSStream", "__version__"]

from livekit.agents import Plugin


class FalPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__)

    def download_files(self):
        pass


Plugin.register_plugin(FalPlugin())
