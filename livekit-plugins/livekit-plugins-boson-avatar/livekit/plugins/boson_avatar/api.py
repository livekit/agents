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
import ipaddress
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
    utils,
)

from .errors import BosonAvatarException
from .log import logger
from .version import __version__

_USER_AGENT = f"livekit-plugins-boson-avatar/{__version__}"
_SESSION_OBJECT = "avatar.livekit.session"


@dataclass(frozen=True)
class AvatarSessionInfo:
    """Hosted Boson Avatar session returned by the provider API."""

    id: str
    avatar_identity: str


class BosonAvatarAPI:
    """Async client for the hosted Boson LiveKit Avatar API."""

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        if not self._api_key:
            raise BosonAvatarException(
                "api_key must be set by passing it to AvatarSession or setting "
                "the BOSON_API_KEY environment variable"
            )

        resolved_url = _resolve_optional_string(api_url, "BOSON_AVATAR_API_URL")
        if not resolved_url:
            raise BosonAvatarException(
                "api_url must be set by passing it to AvatarSession or setting "
                "the BOSON_AVATAR_API_URL environment variable"
            )
        self._api_url = _validate_api_url(resolved_url)
        self._conn_options = conn_options
        self._session = session

    async def start_session(
        self,
        *,
        avatar_id: str,
        livekit_url: str,
        livekit_room: str,
        livekit_token: str,
        avatar_identity: str,
        publisher_identity: str,
        width: int | None = None,
        height: int | None = None,
        max_duration_seconds: int | None = None,
        idempotency_key: str | None = None,
    ) -> AvatarSessionInfo:
        """Start one Boson Avatar participant in an existing LiveKit room."""
        body: dict[str, Any] = {
            "avatar_id": avatar_id,
            "transport": {
                "type": "livekit",
                "url": livekit_url,
                "room_name": livekit_room,
                "participant_token": livekit_token,
                "participant_identity": avatar_identity,
                "publisher_identity": publisher_identity,
                "audio_source": "data_stream",
            },
        }
        if width is not None and height is not None:
            body["output"] = {"width": width, "height": height}
        if max_duration_seconds is not None:
            body["max_duration_seconds"] = max_duration_seconds

        _, data = await self._json(
            "POST",
            "/sessions",
            json=body,
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
            success_statuses=frozenset({200, 201}),
        )
        session_id = data.get("id")
        returned_identity = data.get("avatar_identity")
        response_valid = (
            isinstance(session_id, str)
            and bool(session_id)
            and data.get("object") == _SESSION_OBJECT
            and data.get("status") == "active"
            and returned_identity == avatar_identity
        )
        if not response_valid:
            # A protocol-invalid response can still represent an allocated
            # provider session. Compensate whenever it gives us a usable ID.
            if isinstance(session_id, str) and session_id:
                try:
                    await self.end_session(session_id)
                except Exception:  # noqa: BLE001 - compensation must not hide protocol failure
                    logger.warning(
                        "failed to compensate boson avatar session after invalid response",
                        extra={"session_id": session_id},
                        exc_info=True,
                    )
            if not isinstance(session_id, str) or not session_id:
                raise BosonAvatarException("Boson Avatar API response is missing a session id")
            if returned_identity != avatar_identity:
                raise BosonAvatarException(
                    "Boson Avatar API returned a participant identity that does not match "
                    "the request"
                )
            raise BosonAvatarException("Boson Avatar API returned an invalid active session")

        assert isinstance(session_id, str)
        assert isinstance(returned_identity, str)
        return AvatarSessionInfo(id=session_id, avatar_identity=returned_identity)

    async def end_session(self, session_id: str) -> None:
        """Idempotently stop a hosted Boson Avatar session."""
        status_code, data = await self._json(
            "DELETE",
            f"/sessions/{session_id}",
            allow_empty=True,
            success_statuses=frozenset({200, 204}),
        )
        invalid_response = (status_code == 204 and bool(data)) or (
            status_code == 200
            and (
                data.get("id") != session_id
                or data.get("object") != _SESSION_OBJECT
                or data.get("status") != "terminated"
            )
        )
        if invalid_response:
            raise BosonAvatarException("Boson Avatar API returned an invalid terminated session")

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_empty: bool = False,
        success_statuses: frozenset[int],
    ) -> tuple[int, dict[str, Any]]:
        request_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            **(headers or {}),
        }
        url = f"{self._api_url}{path}"
        error: Exception | None = None

        for attempt in range(self._conn_options.max_retry + 1):
            retry_after: float | None = None
            try:
                async with self._ensure_http_session().request(
                    method,
                    url,
                    json=json,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
                ) as response:
                    payload = await _read_payload(response)
                    if response.status in success_statuses:
                        if payload is None and allow_empty and response.status == 204:
                            return response.status, {}
                        if not isinstance(payload, dict):
                            raise APIStatusError(
                                "Boson Avatar API returned a non-object JSON response",
                                status_code=response.status,
                                body=payload,
                                retryable=False,
                            )
                        return response.status, payload

                    request_id = response.headers.get("x-request-id")
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    if isinstance(payload, dict):
                        error_body = payload.get("error")
                        if isinstance(error_body, dict):
                            request_id = (
                                str(error_body.get("request_id") or request_id or "") or None
                            )
                    raise APIStatusError(
                        "Boson Avatar API returned an error",
                        status_code=response.status,
                        request_id=request_id,
                        body=payload,
                        retryable=not 200 <= response.status < 400,
                    )
            except asyncio.TimeoutError as exc:
                error = APITimeoutError()
                error.__cause__ = exc
            except aiohttp.ClientError as exc:
                error = APIConnectionError()
                error.__cause__ = exc
            except APIStatusError as exc:
                if not exc.retryable:
                    raise
                error = exc

            if attempt == self._conn_options.max_retry:
                break

            logger.warning(
                "boson avatar api request failed, retrying",
                extra={"attempt": attempt + 1, "method": method, "path": path},
            )
            retry_delay = self._conn_options._interval_for_retry(attempt)
            if retry_after is not None:
                retry_delay = max(retry_delay, retry_after)
            await asyncio.sleep(retry_delay)

        raise APIConnectionError("Failed to call Boson Avatar API after all retries.") from error


def _resolve_api_key(value: NotGivenOr[str]) -> str | None:
    if utils.is_given(value) and value:
        return str(value).strip() or None
    env_value = os.getenv("BOSON_API_KEY") or os.getenv("BOSONAI_API_KEY")
    return env_value.strip() if env_value and env_value.strip() else None


def _resolve_optional_string(value: NotGivenOr[str], env_name: str) -> str | None:
    if utils.is_given(value) and value:
        return str(value).strip() or None
    env_value = os.getenv(env_name)
    return env_value.strip() if env_value and env_value.strip() else None


def _validate_api_url(value: str) -> str:
    normalized = value.rstrip("/")
    try:
        parsed = urlsplit(normalized)
        # Accessing port also validates that it is a well-formed integer in range.
        _ = parsed.port
    except ValueError as exc:
        raise BosonAvatarException("api_url must be a valid absolute HTTP(S) base URL") from exc

    if (
        not normalized
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "?" in normalized
        or "#" in normalized
        or any(char.isspace() for char in normalized)
    ):
        raise BosonAvatarException(
            "api_url must be an absolute HTTP(S) base URL without credentials, query, or fragment"
        )

    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise BosonAvatarException(
            "api_url must use HTTPS unless it targets a loopback host; the API key is sent "
            "as a Bearer credential"
        )
    return normalized


def _is_loopback_host(hostname: str) -> bool:
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


async def _read_payload(response: aiohttp.ClientResponse) -> object | None:
    text = await response.text()
    if not text:
        return None
    try:
        payload: object = await response.json(content_type=None)
        return payload
    except ValueError:
        return {"raw": text}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


__all__ = ["AvatarSessionInfo", "BosonAvatarAPI"]
