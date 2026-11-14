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

import json
import os

import aiohttp

from livekit.agents import APIConnectionError, APIError, APIStatusError

from .version import __version__

BASE_URL = "https://api.chariot.in"
API_KEY_HEADER = "chariotai-api-key"
USER_AGENT = f"LiveKit Agents Chariot Plugin/{__version__}"

# The WebSocket close codes the API uses, besides 1000 (normal).
CLOSE_POLICY = 1008  # invalid key or parameters, idle timeout, a backlog limit
CLOSE_SERVER_ERROR = 1011
CLOSE_NO_CREDITS = 4402
CLOSE_SESSION_LIMIT = 4413  # the session reached its maximum length
CLOSE_BUSY = 4429  # every concurrent session the plan allows is in use

# HTTP status for each WebSocket error code, so retries follow the usual 4xx/5xx rules.
_ERROR_STATUS = {
    "bad_request": 400,
    "unknown_context": 400,
    "context_closing": 400,
    "session_closing": 400,
    "text_too_long": 400,
    "insufficient_credits": 402,
    "forbidden": 403,
    "idle_timeout": 408,
    "voice_unavailable": 422,
    "context_limit": 429,
    "concurrency_limit": 429,
    "upstream_error": 502,
}

# 1008 reasons that a fresh connection clears, as opposed to a refusal it would repeat.
_TRANSIENT_POLICY_REASONS = ("idle", "pending audio", "pending commit")


def resolve_api_key(api_key: str | None) -> str:
    api_key = api_key or os.environ.get("CHARIOT_API_KEY")
    if not api_key:
        raise ValueError("Chariot API key required. Set CHARIOT_API_KEY or provide api_key.")
    return api_key


def headers(api_key: str) -> dict[str, str]:
    return {API_KEY_HEADER: api_key, "User-Agent": USER_AGENT}


def ws_base_url(base_url: str) -> str:
    return base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)


def error_event(event: dict) -> APIError:
    """An `error` message on a WebSocket.

    A string `code` names a refused request on a connection that stays open. A numeric `code`
    is the close code of a connection the server is about to close.
    """
    code = event.get("code")
    message = str(event.get("message") or "Chariot API error")
    if isinstance(code, int):
        return close_error(code, message)
    return APIStatusError(
        message, status_code=_ERROR_STATUS.get(code or "", 500), body=code or None
    )


def close_error(code: int | None, reason: str) -> APIError:
    """The error for a connection the server closed before the work was done."""
    message = f"Chariot connection closed ({code}): {reason or 'no reason given'}"
    if code == CLOSE_POLICY and reason.lower().startswith("unauthorized"):
        return APIStatusError(message, status_code=401)
    if code == CLOSE_NO_CREDITS:
        return APIStatusError(message, status_code=402)
    if code == CLOSE_BUSY:
        return APIStatusError(message, status_code=429)
    if code == CLOSE_POLICY and not any(r in reason.lower() for r in _TRANSIENT_POLICY_REASONS):
        return APIStatusError(message, status_code=400)
    return APIConnectionError(message)


async def http_error(resp: aiohttp.ClientResponse, request_id: str | None = None) -> APIStatusError:
    """An error response from an HTTP endpoint, which carries `{"statusCode", "message"}`."""
    text = await resp.text()
    message = text
    try:
        body = json.loads(text)
        if isinstance(body, dict) and body.get("message"):
            message = str(body["message"])
    except ValueError:
        pass
    return APIStatusError(
        message or "Chariot API error",
        status_code=resp.status,
        request_id=request_id,
        body=text or None,
    )
