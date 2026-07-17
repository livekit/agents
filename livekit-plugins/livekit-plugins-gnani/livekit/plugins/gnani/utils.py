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

from __future__ import annotations

import importlib.metadata
from typing import Any


def _websockets_major_version() -> int:
    try:
        version = importlib.metadata.version("websockets")
        return int(version.split(".", 1)[0])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return 13


def ws_header_kwargs(headers: dict[str, str]) -> dict[str, Any]:
    """Return the correct ``connect()`` header kwarg for the installed websockets.

    The new asyncio implementation uses ``additional_headers`` while the legacy
    client uses ``extra_headers``. The top-level ``websockets.connect`` alias
    (used by the STT/TTS clients) only switched to the asyncio implementation in
    websockets 14.0 — in 12.x and 13.x it is still the legacy client. Select the
    kwarg by that boundary so WebSocket STT/TTS works across ``websockets>=12``.
    """
    key = "additional_headers" if _websockets_major_version() >= 14 else "extra_headers"
    return {key: headers}
