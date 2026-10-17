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

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import aiohttp
from dotenv import dotenv_values

from livekit.agents import APIError, APIStatusError


class Configuration:
    """Read only an explicitly selected dotenv file, without changing process environment."""

    def __init__(self, env_file: str | Path | None) -> None:
        path = env_file if env_file is not None else os.environ.get("MICROSOFT_AI_ENV_FILE")
        self._values: dict[str, str | None] = {}
        if path is not None:
            try:
                with open(path, encoding="utf-8") as file:
                    self._values = dict(dotenv_values(stream=file, interpolate=False))
            except (OSError, UnicodeError):
                raise ValueError(
                    "Could not read the selected Microsoft AI environment file"
                ) from None

    def get(self, name: str) -> str | None:
        return os.environ.get(name, self._values.get(name))

    def required(self, value: str | None, name: str) -> str:
        resolved = value if value is not None else self.get(name)
        if resolved is None or not resolved.strip():
            raise ValueError(f"Set {name} or pass its constructor argument")
        return resolved


def positive_timeout(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def status_error(service: str, status: int) -> APIError:
    # Core TTS treats 499 as local cancellation. A remote HTTP 499 is still a failure.
    if status == 499:
        return APIError(f"Microsoft AI {service} returned HTTP 499", retryable=False)
    return APIStatusError(
        f"Microsoft AI {service} request failed",
        status_code=status,
        retryable=status in (408, 429) or 500 <= status < 600,
    )


class HTTPClient:
    """Own a lazy session, or borrow the caller's session without closing it."""

    def __init__(
        self,
        *,
        config: Configuration,
        service: str,
        url: str | None,
        api_key: str | None,
        headers: Mapping[str, str] | None,
        http_session: aiohttp.ClientSession | None,
        auth_header: Literal["Authorization", "api-key"] | None = None,
    ) -> None:
        self.url = config.required(url, f"MICROSOFT_AI_{service}_URL")
        parsed = urlsplit(self.url)
        secure, local = ("wss", "ws") if service == "STT" else ("https", "http")
        if (
            parsed.scheme not in (secure, local)
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(f"MICROSOFT_AI_{service}_URL must be a full {secure} endpoint URL")
        if parsed.scheme == local and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(f"Microsoft AI {service} requires TLS except on loopback endpoints")

        if headers is not None:
            if api_key is not None:
                raise ValueError("Pass either api_key or headers, not both")
            if auth_header is not None:
                raise ValueError("Pass either auth_header or headers, not both")
            # An explicit mapping, including {}, deliberately bypasses API-key environment lookup.
            self.headers = dict(headers)
        else:
            selected_header = "Ocp-Apim-Subscription-Key"
            if service == "STT":
                configured_header = (
                    auth_header
                    if auth_header is not None
                    else config.get("MICROSOFT_AI_STT_AUTH_HEADER")
                )
                selected_header = (
                    configured_header if configured_header is not None else "Authorization"
                )
                if selected_header not in ("Authorization", "api-key"):
                    raise ValueError(
                        "MICROSOFT_AI_STT_AUTH_HEADER/auth_header must be Authorization or api-key"
                    )
            key = config.required(api_key, f"MICROSOFT_AI_{service}_API_KEY")
            if any(ord(char) < 32 or ord(char) == 127 for char in key):
                raise ValueError(
                    f"MICROSOFT_AI_{service}_API_KEY cannot contain control characters"
                )
            self.headers = {
                selected_header: f"Bearer {key}" if selected_header == "Authorization" else key
            }
        self.headers.setdefault("User-Agent", "LiveKit Agents")
        self._session = http_session
        self._owns_session = http_session is None
        self._closed = False

    def session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("Microsoft AI provider is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        self._closed = True
        if self._owns_session and self._session is not None:
            await self._session.close()
