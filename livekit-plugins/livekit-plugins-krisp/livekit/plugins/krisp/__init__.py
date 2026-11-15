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
algorithms. Two authentication backends are supported, exposed under the
``auth`` sub-namespace:

- ``krisp.auth.livekit_cloud()`` (default): LiveKit Cloud-managed auth +
  bundled model. No Krisp env vars required.
- ``krisp.auth.krisp_license(...)``: Krisp-direct auth using a license key
  and a ``.kef`` model file.

The PascalCase classes (:class:`LiveKitCloudAuthProvider`,
:class:`KrispLicenseAuthProvider`) remain available for type annotations.
"""

from livekit.agents import Plugin

from . import auth
from .auth import KrispLicenseAuthProvider, LiveKitCloudAuthProvider
from .log import logger
from .version import __version__
from .viva_filter import (
    KrispVivaFilterFrameProcessor,
    voice_isolation,
    voice_isolation_telephony,
)

__all__ = [
    "KrispVivaFilterFrameProcessor",
    "voice_isolation",
    "voice_isolation_telephony",
    "LiveKitCloudAuthProvider",
    "KrispLicenseAuthProvider",
    "auth",
    "__version__",
]


class KrispPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(KrispPlugin())
