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

import aiohttp

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, utils

from .errors import ErrorType, SynthesiaError
from .log import logger
from .types import StartSessionRequest, StartSessionResponse

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


class SynthesiaAPI:
    """Async client for the Synthesia interactive-avatar session API.

    Opens one session per ``start_session`` call. Pass ``session`` to reuse an
    ``aiohttp.ClientSession``; when omitted the LiveKit Agents shared session is
    used.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")
        self._session = session
        self._conn_options = conn_options

    def _http(self) -> aiohttp.ClientSession:
        return self._session or utils.http_context.http_session()

    async def start_session(
        self,
        request: StartSessionRequest,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> StartSessionResponse:
        conn_options = conn_options or self._conn_options
        url = self._api_url + SESSION_PATH
        headers = {"Authorization": self._api_key}
        payload: dict[str, object] = {
            "avatarIds": [_public_avatar_id(a) for a in request.avatar_ids],
            "livekitUrl": request.livekit_url,
            "livekitToken": request.lk_token,
        }
        timeout = aiohttp.ClientTimeout(total=conn_options.timeout)

        last_status: int | None = None
        last_body: object = None
        last_exc: Exception | None = None

        for attempt in range(conn_options.max_retry + 1):
            status: int | None = None
            try:
                async with self._http().post(
                    url, headers=headers, json=payload, timeout=timeout
                ) as resp:
                    status = resp.status
                    body = await _read_json(resp)
                    if resp.ok:
                        return self._parse_success(body)
                    if status < 500:
                        raise self._mapped_error(resp, body)
                    last_status, last_body, last_exc = status, body, None
                    logger.debug(
                        "synthesia session request failed, retrying",
                        extra={"status": status},
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # A 5xx whose body fails to arrive is still a 5xx, not a transport failure.
                last_status = status if status is not None and status >= 500 else None
                last_body, last_exc = None, e
                logger.debug(
                    "synthesia session request errored, retrying",
                    extra={"error": type(e).__name__},
                )

            if attempt < conn_options.max_retry:
                await asyncio.sleep(conn_options.retry_interval)

        raise SynthesiaError(
            _exhausted_message(conn_options.max_retry + 1, last_status, last_body),
            type=ErrorType.CONNECTION,
            body=last_body,
            status=last_status,
            request_id=_body_request_id(last_body),
        ) from last_exc

    def _parse_success(self, body: object) -> StartSessionResponse:
        if isinstance(body, dict):
            session_id = body.get("id") or body.get("session_id")
            if isinstance(session_id, str) and session_id:
                return StartSessionResponse(session_id=session_id)
        raise SynthesiaError("Synthesia response did not contain a session id")

    def _mapped_error(self, resp: aiohttp.ClientResponse, body: object) -> SynthesiaError:
        code = _body_code(body)
        error_type = _CODE_TO_ERROR.get(code or "")
        if error_type is None:
            # A problem body names its code; an unknown one is a new code, not a hint to
            # guess the type from the status.
            error_type = (
                None
                if _is_problem(body) and code is not None
                else _STATUS_TO_ERROR.get(resp.status)
            )
        message = _body_message(body) or (
            f"Synthesia request failed ({code or f'HTTP {resp.status}'})"
        )
        request_id = _body_request_id(body)
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
            request_id=request_id,
        )


def _exhausted_message(attempts: int, status: int | None, body: object) -> str:
    message = "could not start a Synthesia session"
    if attempts > 1:
        message = f"{message} after {attempts} attempts"
    if status is None:
        return f"{message}; last attempt: connection error"
    message = f"{message}; last attempt: HTTP {status}"
    detail = _body_message(body)
    if detail is not None:
        message = f"{message}: {detail}"
    request_id = _body_request_id(body)
    if request_id is not None:
        message = f"{message} (request {request_id})"
    return message


def _is_problem(body: object) -> bool:
    return isinstance(body, dict) and isinstance(body.get("type"), str)


def _body_code(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    code = err.get("code") if isinstance(err, dict) else None
    code = code or body.get("code")
    return code if isinstance(code, str) else None


def _body_message(body: object) -> str | None:
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


def _body_request_id(body: object) -> str | None:
    if isinstance(body, dict):
        request_id = body.get("requestId")
        if isinstance(request_id, str):
            return request_id
    return None


def _public_avatar_id(avatar_id: str) -> str:
    """Prefix a gallery id for the wire, leaving an already-prefixed id alone.

    Callers pass the raw gallery id they see in Synthesia, and the API takes the
    namespaced form.
    """
    return avatar_id if avatar_id.startswith(_AVATAR_ID_PREFIX) else _AVATAR_ID_PREFIX + avatar_id


def _parse_retry_after(resp: aiohttp.ClientResponse, body: object) -> float | None:
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


async def _read_json(resp: aiohttp.ClientResponse) -> object:
    # Errors arrive as application/problem+json, which aiohttp refuses by default,
    # so the content type is checked here rather than delegated.
    try:
        return await resp.json(content_type=resp.content_type)
    except (aiohttp.ContentTypeError, ValueError):
        return None
