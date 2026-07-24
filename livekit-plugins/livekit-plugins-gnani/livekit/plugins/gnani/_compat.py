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

    websockets >= 13 renamed ``extra_headers`` to ``additional_headers``. Support
    both so WebSocket STT/TTS works when another dependency pins websockets < 13.
    """
    key = "additional_headers" if _websockets_major_version() >= 13 else "extra_headers"
    return {key: headers}
