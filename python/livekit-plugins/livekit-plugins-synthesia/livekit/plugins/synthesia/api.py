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

from pydantic import JsonValue, TypeAdapter

from livekit.agents import http

from .errors import ErrorType, SynthesiaError

_JSON_VALUE = TypeAdapter[JsonValue](JsonValue)

SESSION_PATH = "/api/interactive-avatars/sessions"
_AVATAR_ID_PREFIX = "av_"

# Machine-readable error codes returned by the backend in the response body.
_CODE_TO_ERROR: dict[str, ErrorType] = {
    "unauthorized": ErrorType.AUTH,
    "invalid_api_key": ErrorType.AUTH,
    "unknown_avatar": ErrorType.UNKNOWN_AVATAR,
    "avatar_not_accessible": ErrorType.UNKNOWN_AVATAR,
    "quota_exceeded": ErrorType.QUOTA_EXCEEDED,
    "rate_limited": ErrorType.RATE_LIMITED,
    "invalid_token": ErrorType.INVALID_ROOM_TOKEN,
    "invalid_livekit_credentials": ErrorType.LIVEKIT_CREDENTIALS_REJECTED,
    "validation_error": ErrorType.INVALID_SESSION_REQUEST,
    "bad_request": ErrorType.INVALID_SESSION_REQUEST,
    "unknown_reference": ErrorType.UNKNOWN_AVATAR,
    "unauthenticated": ErrorType.AUTH,
    "forbidden": ErrorType.AUTH,
    "insufficient_scope": ErrorType.AUTH,
    "not_authorized": ErrorType.AUTH,
    "feature_not_in_plan": ErrorType.FEATURE_NOT_IN_PLAN,
    "payment_required": ErrorType.QUOTA_EXCEEDED,
    "concurrency_limit": ErrorType.CONCURRENCY_LIMIT,
    # Production still emits this until the taxonomy rename reaches prod. Drop
    # once the API only sends "concurrency_limit".
    "concurrency_limit_exceeded": ErrorType.CONCURRENCY_LIMIT,
}

# Fallback when the body carries no code. A bare 401 is ambiguous
# between the two credentials in play, so it stays the broader auth error.
_STATUS_TO_ERROR: dict[int, ErrorType] = {
    401: ErrorType.AUTH,
    403: ErrorType.AUTH,
    402: ErrorType.QUOTA_EXCEEDED,
    404: ErrorType.UNKNOWN_AVATAR,
    429: ErrorType.RATE_LIMITED,
}


def session_error(resp: http.HttpResponse, body: JsonValue) -> SynthesiaError:
    """The error a refused session request names, by its body's code, else its status."""
    code = _body_code(body)
    error_type = _CODE_TO_ERROR.get(code or "")
    if error_type is None:
        # A problem body names its code; an unknown one is a new code, not a hint to
        # guess the type from the status.
        error_type = (
            None if _is_problem(body) and code is not None else _STATUS_TO_ERROR.get(resp.status)
        )
    message = _body_message(body) or (f"Synthesia request failed ({code or f'HTTP {resp.status}'})")
    retry_after = (
        _parse_retry_after(resp, body)
        if error_type in (ErrorType.RATE_LIMITED, ErrorType.CONCURRENCY_LIMIT)
        else None
    )
    return SynthesiaError(
        message,
        type=error_type,
        retry_after=retry_after,
        body=body,
        status=resp.status,
        request_id=_body_request_id(body),
    )


def _is_problem(body: JsonValue) -> bool:
    return isinstance(body, dict) and isinstance(body.get("type"), str)


def _body_code(body: JsonValue) -> str | None:
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    code = err.get("code") if isinstance(err, dict) else None
    code = code or body.get("code")
    return code if isinstance(code, str) else None


def _body_message(body: JsonValue) -> str | None:
    if not isinstance(body, dict):
        return None
    if _is_problem(body):
        candidates = (body.get("detail"), body.get("title"))
        return next((c for c in candidates if isinstance(c, str) and c), None)
    err = body.get("error")
    message = err.get("message") if isinstance(err, dict) else None
    # Some responses use a flat shape, e.g.
    # {"error": "Forbidden", "context": "User is not authenticated"}.
    detail = message or body.get("message") or body.get("context")
    if detail is None:
        return err if isinstance(err, str) else None
    if isinstance(detail, str):
        return detail
    detail = json.dumps(detail, separators=(", ", ": "))
    code = _body_code(body)
    return f"{code}: {detail}" if code else detail


def _body_request_id(body: JsonValue) -> str | None:
    if isinstance(body, dict):
        request_id = body.get("requestId")
        if isinstance(request_id, str):
            return request_id
    return None


def public_avatar_id(avatar_id: str) -> str:
    """Prefix a gallery id for the wire, leaving an already-prefixed id alone.

    Callers pass the raw gallery id they see in Synthesia, and the API takes the
    namespaced form.
    """
    return avatar_id if avatar_id.startswith(_AVATAR_ID_PREFIX) else _AVATAR_ID_PREFIX + avatar_id


def _parse_retry_after(resp: http.HttpResponse, body: JsonValue) -> float | None:
    header = resp.headers.get("Retry-After")
    if header is not None:
        try:
            return float(header)
        except ValueError:
            pass
    if isinstance(body, dict):
        candidates = [body.get("retry_after")]
        err = body.get("error")
        if isinstance(err, dict):
            candidates.append(err.get("retry_after"))
        for value in candidates:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def parse_json(body: bytes) -> JsonValue:
    try:
        return _JSON_VALUE.validate_json(body)
    except ValueError:
        return None
