# Copyright 2026 LiveKit, Inc.
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

import asyncio
import json
import uuid
from typing import Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
)

from .log import logger
from .version import __version__

DEFAULT_API_URL = "https://api.humanlike.co"
SESSIONS_PATH = "/v1/avatar/sessions"


class HumanlikeException(Exception):
    """A configuration error, raised before any request is sent."""


class HumanlikeAPI:
    """Client for the Humanlike avatar session API.

    ``POST /v1/avatar/sessions`` starts an avatar, ``POST /v1/avatar/sessions/{id}/renew``
    starts the successor of a capped session and ``DELETE /v1/avatar/sessions/{id}``
    ends one, which stops its billing.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        *,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")
        self._session = session
        self._conn_options = conn_options

    async def create_session(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Start an avatar. Answers once the avatar backend is up, which can take a cold
        GPU boot, so ``timeout`` covers the whole request."""
        return await self._request(
            "POST",
            SESSIONS_PATH,
            body=payload,
            timeout=timeout,
            idempotency_key=str(uuid.uuid4()),
        )

    async def renew_session(
        self, session_id: str, payload: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        """Start the successor of a capped session; it joins with the same identity."""
        return await self._request(
            "POST",
            f"{SESSIONS_PATH}/{session_id}/renew",
            body=payload,
            timeout=timeout,
            idempotency_key=str(uuid.uuid4()),
        )

    async def end_session(self, session_id: str, *, timeout: float = 15.0) -> None:
        """End a session. Ending one that already ended is not an error."""
        await self._request("DELETE", f"{SESSIONS_PATH}/{session_id}", timeout=timeout)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"livekit-plugins-humanlike/{__version__}",
        }
        # The same key on every attempt: a retried start returns the session the first
        # attempt created instead of starting a second avatar.
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key

        url = f"{self._api_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self._conn_options.max_retry + 1):
            try:
                async with self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(
                        total=timeout, sock_connect=self._conn_options.timeout
                    ),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            _error_message(response.status, text),
                            status_code=response.status,
                            body=text,
                        )
                    if response.status == 204:
                        return {}
                    try:
                        data = await response.json(content_type=None)
                    except ValueError as e:
                        raise APIConnectionError(
                            "Humanlike API returned a malformed response", retryable=False
                        ) from e
                    return data if isinstance(data, dict) else {}
            except APIStatusError as e:
                if not e.retryable:
                    raise
                last_error = e
            except APIConnectionError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e

            if attempt < self._conn_options.max_retry:
                logger.warning(
                    "Humanlike API request failed, retrying",
                    extra={"method": method, "path": path, "error": str(last_error)},
                )
                await asyncio.sleep(self._conn_options._interval_for_retry(attempt))

        if isinstance(last_error, APIStatusError):
            raise last_error
        raise APIConnectionError(
            f"Failed to call the Humanlike API at {url} after all retries"
        ) from last_error


def _error_message(status: int, body: str) -> str:
    """The gateway answers errors as ``{"error": {"code", "message"}}``."""
    try:
        error = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    if not isinstance(error, dict):
        error = {}
    message = error.get("message") or f"Humanlike API returned HTTP {status}"
    code = error.get("code")
    return f"{message} ({code})" if code else message
