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

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import JsonValue, TypeAdapter

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    http,
    utils,
)

from .errors import BosonAvatarException
from .version import __version__

_JSON_VALUE = TypeAdapter[JsonValue](JsonValue)

_USER_AGENT = f"livekit-plugins-boson-avatar/{__version__}"
_SESSION_OBJECT = "avatar.livekit.session"


@dataclass(frozen=True)
class AvatarInfo:
    """Avatar that the authenticated Boson project may render."""

    avatar_id: str
    name: str


class BosonAvatarAPI:
    """Async client for the hosted Boson LiveKit Avatar API."""

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: http.HttpSession | None = None,
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

    async def list_avatars(self) -> list[AvatarInfo]:
        """List Avatars available to the authenticated Boson project."""
        _, payload = await http.retry_request(
            lambda: self.request("GET", "/avatars", success_statuses=frozenset({200})),
            conn_options=self._conn_options,
        )
        raw_avatars = payload.get("data")
        if payload.get("object") != "avatar.list" or not isinstance(raw_avatars, list):
            raise BosonAvatarException("Boson Avatar API returned an invalid Avatar list")

        avatars: list[AvatarInfo] = []
        seen_ids: set[str] = set()
        for raw_avatar in raw_avatars:
            if not isinstance(raw_avatar, dict):
                raise BosonAvatarException("Boson Avatar API returned an invalid Avatar list")
            avatar_id = raw_avatar.get("avatar_id")
            name = raw_avatar.get("name")
            if not isinstance(avatar_id, str) or not isinstance(name, str):
                raise BosonAvatarException("Boson Avatar API returned an invalid Avatar list")
            avatar_id = avatar_id.strip()
            name = name.strip()
            if not avatar_id or not name or avatar_id in seen_ids:
                raise BosonAvatarException("Boson Avatar API returned an invalid Avatar list")
            seen_ids.add(avatar_id)
            avatars.append(AvatarInfo(avatar_id=avatar_id, name=name))
        return avatars

    def _ensure_http_session(self) -> http.HttpSession:
        if self._session is None:
            self._session = http.http_session()
        return self._session

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, JsonValue] | None = None,
        headers: dict[str, str] | None = None,
        allow_empty: bool = False,
        success_statuses: frozenset[int],
    ) -> tuple[int, dict[str, JsonValue]]:
        """One request; a status outside ``success_statuses`` or a body that is not an object
        breaks the provider's contract."""
        request_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            **(headers or {}),
        }
        session = self._ensure_http_session().with_request_id("x-request-id")
        async with session.request(
            method,
            f"{self._api_url}{path}",
            json=json,
            headers=request_headers,
            conn_options=self._conn_options,
        ) as response:
            try:
                await response.raise_for_status()
            except APIStatusError as exc:
                exc.request_id = _error_request_id(exc.body) or exc.request_id
                raise
            if response.status not in success_statuses:
                raise APIStatusError(
                    "Boson Avatar API returned an unexpected status",
                    status_code=response.status,
                    kind="protocol",
                )
            payload = await _read_payload(response)
            if payload is None and allow_empty and response.status == 204:
                return response.status, {}
            if not isinstance(payload, dict):
                raise APIStatusError(
                    "Boson Avatar API returned a non-object JSON response",
                    status_code=response.status,
                    body=payload,
                    kind="protocol",
                    detail=str(payload),
                )
            return response.status, payload


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
    except ValueError:
        raise BosonAvatarException("api_url must be a valid absolute HTTP(S) base URL") from None

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
    hostname = hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


async def list_avatars(
    *,
    api_key: NotGivenOr[str] = NOT_GIVEN,
    api_url: NotGivenOr[str] = NOT_GIVEN,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
) -> list[AvatarInfo]:
    """List project Avatars using the configured provider endpoint."""
    session = http.HttpSession()
    try:
        return await BosonAvatarAPI(
            api_key=api_key,
            api_url=api_url,
            conn_options=conn_options,
            session=session,
        ).list_avatars()
    finally:
        await session.aclose()


async def _read_payload(response: http.HttpResponse) -> JsonValue:
    text = await response.text()
    if not text:
        return None
    try:
        payload = _JSON_VALUE.validate_python(await response.json())
        return payload
    except ValueError:
        return {"raw": text}


def _error_request_id(body: object) -> str | None:
    """The request id Boson puts in its error body."""
    try:
        payload = _JSON_VALUE.validate_json(str(body))
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    request_id = error.get("request_id") if isinstance(error, dict) else None
    return str(request_id) if request_id else None


__all__ = ["AvatarInfo", "BosonAvatarAPI", "list_avatars"]
