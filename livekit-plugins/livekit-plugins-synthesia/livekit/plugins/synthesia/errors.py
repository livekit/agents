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

import enum
import json

from livekit.agents import APIError

__all__ = [
    "ErrorType",
    "SynthesiaError",
]


class ErrorType(enum.Enum):
    """What went wrong, for callers that want to branch on it."""

    AUTH = "auth"
    """The API key is invalid, expired, or lacks the scope this endpoint requires."""

    FEATURE_NOT_IN_PLAN = "feature_not_in_plan"
    """The workspace's plan does not include interactive avatars."""

    INVALID_ROOM_TOKEN = "invalid_room_token"
    """The room token cannot produce a joined session.

    Distinct from ``AUTH``, which concerns the Synthesia API key. Raised for a
    token that is malformed or missing the attribute naming the agent the
    avatar publishes audio for.
    """

    LIVEKIT_CREDENTIALS_REJECTED = "livekit_credentials_rejected"
    """The LiveKit project the token was signed for would not accept it.

    The token itself is well formed, so the API key and secret it was minted
    with are the thing to check, not the Synthesia credentials.
    """

    INVALID_SESSION_REQUEST = "invalid_session_request"
    """The backend rejected the session request payload."""

    UNKNOWN_AVATAR = "unknown_avatar"
    """Avatar is not in the gallery or not accessible to the workspace."""

    QUOTA_EXCEEDED = "quota_exceeded"
    """The workspace's session quota is exhausted."""

    RATE_LIMITED = "rate_limited"
    """The request was throttled. ``retry_after`` may carry the back-off."""

    CONCURRENCY_LIMIT = "concurrency_limit"
    """Every concurrent-session slot is in use.

    The API sends no ``Retry-After`` for this today, so ``retry_after`` is
    normally ``None``; it is honoured if one ever arrives.
    """

    TIMEOUT = "timeout"
    """The avatar did not join within ``join_timeout``."""

    CONNECTION = "connection"
    """No usable response after retries.

    The last attempt either failed to connect or was answered with a 5xx;
    ``status`` and ``request_id`` describe that answer when there was one.
    """


_RETRYABLE_TYPES = {
    ErrorType.RATE_LIMITED,
    ErrorType.CONCURRENCY_LIMIT,
    ErrorType.TIMEOUT,
    ErrorType.CONNECTION,
}


class SynthesiaError(APIError):
    """Every error the Synthesia plugin raises. Also a :class:`~livekit.agents.APIError`.

    - ``type`` identifies what went wrong; see :class:`ErrorType`.
    - ``retryable`` reports whether retrying the same call could plausibly
      succeed, and defaults based on ``type`` unless overridden.
    - ``retry_after`` carries the server-provided back-off in seconds for a
      ``RATE_LIMITED`` or ``CONCURRENCY_LIMIT`` error, when the backend
      supplied one.
    - ``status`` is the HTTP status the API answered, or ``None`` when no
      answer arrived.
    - ``request_id`` is the API's ``requestId`` when the body carried one.
      Quote it when contacting support.
    """

    def __init__(
        self,
        message: str,
        *,
        type: ErrorType | None = None,
        body: object | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        if not isinstance(message, str):
            message = _stringify(message)
        self.type = type
        self.retry_after = retry_after
        self.status = status
        self.request_id = request_id
        if retryable is None:
            retryable = type in _RETRYABLE_TYPES
        super().__init__(message, body=body, retryable=retryable)


def _stringify(value: object) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, separators=(", ", ": "))
        except (TypeError, ValueError):
            pass
    return str(value)
