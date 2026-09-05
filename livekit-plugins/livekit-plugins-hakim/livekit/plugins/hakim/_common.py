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

"""Shared wire-protocol helpers for the Hakim plugin.

Both `stt.py` and `tts.py` talk directly to Hakim's public REST/WebSocket
API (see https://tryhakim.ai/docs) over plain `websockets` / `aiohttp`
connections. This module exists only to avoid duplicating the handful of
things both need: region -> host resolution, the `ws(s)://` upgrade URL,
the auth header, and a shared exception type for the server's `error`
frame.
"""

from __future__ import annotations

import os
from typing import Literal

Region = Literal["auto", "de", "uae", "ksa"]

_REGION_HOSTS: dict[str, str] = {
    "auto": "api.tryhakim.ai",
    "de": "de.api.tryhakim.ai",
    "uae": "uae.api.tryhakim.ai",
    "ksa": "ksa.api.tryhakim.ai",
}


def resolve_api_key(explicit: str | None) -> str:
    key = explicit or os.environ.get("HAKIM_API_KEY")
    if not key:
        raise ValueError(
            "Hakim API key not set. Pass api_key=... or set the HAKIM_API_KEY "
            "environment variable. Get a key from the Hakim dashboard "
            "(Settings -> API keys)."
        )
    return key


def resolve_ws_url(path: str, *, region: Region = "auto", base_url: str | None = None) -> str:
    """Build the `wss://` upgrade URL for a realtime endpoint.

    `base_url` (if given) wins outright -- it's how staging/self-hosted
    deployments override the default region hosts. Otherwise the region
    code maps to one of Hakim's published regional endpoints.
    """
    if base_url:
        host = (
            base_url.removeprefix("https://")
            .removeprefix("http://")
            .removeprefix("wss://")
            .removeprefix("ws://")
        )
        host = host.rstrip("/")
    else:
        host = _REGION_HOSTS.get(region, _REGION_HOSTS["auto"])
    return f"wss://{host}{path}"


def resolve_http_url(path: str, *, region: Region = "auto", base_url: str | None = None) -> str:
    if base_url:
        host = base_url.removeprefix("https://").removeprefix("http://").rstrip("/")
    else:
        host = _REGION_HOSTS.get(region, _REGION_HOSTS["auto"])
    return f"https://{host}{path}"


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


class HakimStreamError(Exception):
    """Raised when a Hakim realtime session sends an `error` frame.

    Mirrors the `code` / `message` / `retryable` fields on the `error`
    frame documented at https://tryhakim.ai/docs so callers can branch on
    `code` without parsing the raw frame themselves.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.retryable = retryable
