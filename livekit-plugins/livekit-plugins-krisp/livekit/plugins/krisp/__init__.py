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

"""Krisp VIVA plugin for LiveKit Agents

This plugin provides real-time noise reduction using Krisp's proprietary
algorithms. Two authentication backends are supported:

- :class:`LiveKitCloudAuthProvider` (default): LiveKit Cloud-managed auth +
  bundled model. No Krisp env vars required.
- :class:`KrispLicenseAuthProvider`: Krisp-direct auth using a license key
  and a ``.kef`` model file.
"""

from livekit.agents import Plugin

from .auth import (
    KrispAuthProvider,
    KrispBackend,
    KrispLicenseAuthProvider,
    LiveKitCloudAuthProvider,
)
from .krisp_instance import KrispSDKManager
from .log import logger
from .version import __version__
from .viva_filter import KrispVivaFilterFrameProcessor

__all__ = [
    "KrispVivaFilterFrameProcessor",
    "KrispSDKManager",
    "KrispAuthProvider",
    "KrispBackend",
    "LiveKitCloudAuthProvider",
    "KrispLicenseAuthProvider",
    "__version__",
]


class KrispPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(KrispPlugin())
