# Copyright 2026 Atmanity
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

"""Thin async client for the Atmee API used by the plugin.

Two resources matter here: **avatars** — Atmee v1 avatars, i.e. talking heads
generated from a single portrait — (create one from a portrait, read its
status) and **avatar sessions** (render an avatar into your LiveKit room for
your agent, end it). Authentication is your Atmee API key (``sk_atmee_...``)
in the ``X-Api-Key`` header; the key never reaches a browser.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    utils,
)

from .log import logger

DEFAULT_API_URL = "https://api.atmanity.us"

WaitFor = Literal["initializing", "avatar_joined"]

AvatarVersion = Literal["v1", "v2"]
"""An Atmee avatar generation.

``"v1"`` is a talking head generated from a single portrait, lip-synced to the
agent's speech: the avatars this plugin renders today. ``"v2"`` is the name of
Atmee's next avatar generation; it is not available through this plugin yet
and requesting it raises :class:`ValueError`.
"""

SUPPORTED_AVATAR_VERSIONS: frozenset[AvatarVersion] = frozenset({"v1"})
"""The avatar versions this release of the plugin can render."""

# Overall budget of the start call: the worker acknowledges within seconds, but
# waiting for the avatar to join covers model priming on a cold pod (the
# server's own join timeout is 90 s) plus the room join.
_START_TOTAL_TIMEOUT: dict[str, float] = {"initializing": 60.0, "avatar_joined": 180.0}
_DEFAULT_TOTAL_TIMEOUT = 120.0


class AtmeeException(Exception):
    """An Atmee API call failed.

    ``status_code`` is the HTTP status (0 when the request never got an
    answer), ``code`` the machine-readable ``error`` field of the response
    body when the API sent one (``invalid_livekit_token``,
    ``avatar_not_renderable``, ``insufficient_credits``, ...), and
    ``message`` its human-readable text.
    """

    def __init__(self, message: str, *, status_code: int = 0, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[{self.code}]")
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        return " ".join(parts)


class AtmeeNoCapacityError(AtmeeException):
    """Every rendering worker is busy (HTTP 503 ``no_capacity``).

    Never retried by the client: a render slot frees up on the order of
    seconds to minutes, not the milliseconds a transport retry covers.
    ``retry_after`` is the server's hint in seconds, when it sent one.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=503, code="no_capacity")
        self.retry_after = retry_after


class AtmeeAvatarNotReadyError(AtmeeException):
    """The avatar cannot be rendered yet (HTTP 409 ``avatar_not_renderable``):
    it has no portrait, or its appearance is still being processed."""


def _check_avatar_version(avatar_version: str) -> None:
    """Reject an avatar version this plugin cannot render, before any API call."""
    if avatar_version in SUPPORTED_AVATAR_VERSIONS:
        return
    supported = ", ".join(f"'{v}'" for v in sorted(SUPPORTED_AVATAR_VERSIONS))
    raise ValueError(
        f"avatar_version '{avatar_version}' is not supported by livekit-plugins-atmee yet; "
        f"only {supported} (talking-head avatars rendered from a single portrait) "
        "can be rendered today"
    )


@dataclass
class AvatarSessionInfo:
    """The ``202`` body of ``POST /v1/avatars/{avatarId}/avatar_sessions``.

    ``avatar_version`` is the avatar generation the session was requested for
    (``"v1"``); the plugin fills it in, the API body carries no such field.
    """

    session_id: str
    status: str
    avatar_participant_identity: str
    agent_identity: str
    room_name: str
    max_duration_seconds: int
    billing_mode: str
    avatar_version: str = "v1"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AvatarInfo:
    """What ``GET /v1/avatars/{avatarId}`` (and a create) report about an avatar.

    ``kind`` (``conversational`` | ``render_only``) says whether the avatar has
    a voice and persona; ``version`` is the avatar generation (``"v1"``, filled
    in by the plugin: the API reports no version).
    """

    avatar_id: str
    status: str
    kind: str | None = None
    name: str | None = None
    version: str = "v1"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        return self.status == "ready"


_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class AtmeeAPI:
    """Async client for the Atmee API.

    Args:
        api_key: your ``sk_atmee_...`` key; defaults to ``ATMEE_API_KEY``.
        api_url: API base; defaults to ``ATMEE_API_URL`` or ``https://api.atmanity.us``.
        conn_options: retry/timeout policy for transport failures and 5xx
            answers (a 503 ``no_capacity`` is never retried).
        session: an ``aiohttp.ClientSession`` to reuse. Inside a LiveKit job the
            worker's shared session is used; outside (scripts, tests) a private
            one is created and closed by :meth:`aclose`.
    """

    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        key = api_key or os.getenv("ATMEE_API_KEY")
        if not key:
            raise AtmeeException(
                "ATMEE_API_KEY must be set (or pass api_key=...); create one at "
                "https://www.atmee.ai/studio/api-keys"
            )
        self._api_key = key
        self._api_url = (api_url or os.getenv("ATMEE_API_URL") or DEFAULT_API_URL).rstrip("/")
        _check_api_url(self._api_url)
        self._conn_options = conn_options
        self._session = session
        self._owns_session = False

    @property
    def api_url(self) -> str:
        return self._api_url

    # --- avatar sessions -------------------------------------------------

    async def create_avatar_session(
        self,
        avatar_id: str,
        *,
        livekit_url: str,
        livekit_token: str,
        agent_identity: str | None = None,
        max_duration_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        wait_for: WaitFor = "initializing",
        avatar_version: AvatarVersion = "v1",
    ) -> AvatarSessionInfo:
        """Render ``avatar_id`` into the room ``livekit_token`` grants.

        ``livekit_token`` is a LiveKit access token you minted with YOUR
        project's secret for the avatar participant: ``kind=agent``, a
        ``roomJoin`` grant for your room, and the ``lk.publish_on_behalf``
        attribute naming your agent (:class:`AvatarSession` does this for
        you). Returns as soon as the rendering worker acknowledged the start,
        or once the avatar joined with ``wait_for="avatar_joined"``.

        ``avatar_version`` names the avatar generation to render; only
        ``"v1"`` is available today and anything else raises ``ValueError``
        before the request is sent. It is plugin-side only: the request body
        carries no version field.
        """
        _check_avatar_version(avatar_version)
        payload: dict[str, Any] = {"livekitUrl": livekit_url, "livekitToken": livekit_token}
        if agent_identity:
            payload["agentIdentity"] = agent_identity
        if max_duration_seconds is not None:
            payload["maxDurationSeconds"] = int(max_duration_seconds)
        if metadata:
            payload["metadata"] = metadata
        params: dict[str, str] | None = (
            {"waitFor": wait_for} if wait_for != "initializing" else None
        )
        # Never retried: the POST reserves and bills a session as soon as the
        # server accepts it, so a retry after a timeout or a 5xx could open a
        # second render for the same room. The caller decides whether to try
        # again (the AvatarSession does not).
        data = await self._request(
            "POST",
            f"/v1/avatars/{avatar_id}/avatar_sessions",
            json=payload,
            params=params,
            retry=False,
            total_timeout=_START_TOTAL_TIMEOUT[wait_for],
        )
        session_id = data.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise AtmeeException(
                f"the Atmee API accepted the avatar session but returned no sessionId: {data!r}",
                code="invalid_response",
            )
        return AvatarSessionInfo(
            session_id=session_id,
            status=str(data.get("status", "")),
            avatar_participant_identity=str(data.get("avatarParticipantIdentity", "")),
            agent_identity=str(data.get("agentIdentity", "")),
            room_name=str(data.get("roomName", "")),
            max_duration_seconds=int(data.get("maxDurationSeconds") or 0),
            billing_mode=str(data.get("billingMode", "")),
            avatar_version=avatar_version,
            raw=data,
        )

    async def end_avatar_session(self, session_id: str) -> dict[str, Any]:
        """Stop billing for a session now. Idempotent: ending an ended session
        answers normally with ``alreadyEnded: true``. Does not remove the avatar
        from your room — :meth:`AvatarSession.aclose` does that."""
        return await self._request("POST", f"/v1/avatar_sessions/{session_id}/end")

    async def get_avatar_session(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/avatar_sessions/{session_id}")

    # --- avatars -----------------------------------------------------------

    async def create_avatar(
        self,
        name: str,
        image: str | Path | bytes,
        *,
        description: str | None = None,
        content_type: str | None = None,
        avatar_version: AvatarVersion = "v1",
    ) -> str:
        """Create a render-only **v1 avatar** from a single portrait and return its id.

        ``image`` is a local file path, the image bytes, or an ``https://``
        URL the Atmee service can download. A portrait-only avatar is ready
        at once; nothing to poll. Add a voice later in the Atmee studio (or
        ``PUT /v1/avatars/{id}/voice``) to make it conversational as well.

        ``avatar_version`` is the avatar generation to create; only ``"v1"``
        is available today and anything else raises ``ValueError`` before any
        request is sent. The API contract has no version field, so nothing is
        added to the request.
        """
        info = await self.create_avatar_info(
            name,
            image,
            description=description,
            content_type=content_type,
            avatar_version=avatar_version,
        )
        return info.avatar_id

    async def create_avatar_info(
        self,
        name: str,
        image: str | Path | bytes,
        *,
        description: str | None = None,
        content_type: str | None = None,
        avatar_version: AvatarVersion = "v1",
    ) -> AvatarInfo:
        """Like :meth:`create_avatar`, returning the whole response."""
        _check_avatar_version(avatar_version)
        if isinstance(image, str) and _is_url(image):
            manifest: dict[str, Any] = {
                "schemaVersion": 1,
                "name": name,
                "assets": {"image": {"url": image}},
            }
            if description:
                manifest["description"] = description
            data = await self._request("POST", "/v1/avatars", json=manifest, retry=False)
            return _avatar_info(data, version=avatar_version)

        if isinstance(image, bytes | bytearray):
            blob = bytes(image)
            filename = "portrait" + _extension_for(content_type) if content_type else "portrait.jpg"
            ctype = content_type or "image/jpeg"
        else:
            blob, filename, ctype = await asyncio.to_thread(_load_image_file, image, content_type)
        form = aiohttp.FormData()
        form.add_field("name", name)
        if description:
            form.add_field("description", description)
        form.add_field("file", blob, filename=filename, content_type=ctype)
        data = await self._request("POST", "/v1/avatars", data=form, retry=False)
        return _avatar_info(data, version=avatar_version)

    async def get_avatar(self, avatar_id: str) -> AvatarInfo:
        return await self._get_avatar(avatar_id)

    async def _get_avatar(self, avatar_id: str, *, deadline: float | None = None) -> AvatarInfo:
        return _avatar_info(
            await self._request("GET", f"/v1/avatars/{avatar_id}", deadline=deadline)
        )

    async def wait_until_ready(
        self, avatar_id: str, *, timeout: float = 600.0, poll_interval: float = 5.0
    ) -> AvatarInfo:
        """Poll until the avatar reports ``ready`` (a conversational avatar
        builds asynchronously; a portrait-only one is ready immediately)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        status = "unknown"
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AtmeeException(
                    f"avatar {avatar_id} not ready after {timeout:.0f}s (status {status})",
                    code="timeout",
                )
            try:
                info = await self._get_avatar(avatar_id, deadline=deadline)
            except AtmeeException as e:
                if loop.time() >= deadline and e.status_code == 0:
                    raise AtmeeException(
                        f"avatar {avatar_id} not ready after {timeout:.0f}s (status {status})",
                        code="timeout",
                    ) from e
                raise
            status = info.status
            if info.ready:
                return info
            if info.status == "failed":
                raise AtmeeException(f"avatar {avatar_id} failed to build", code="avatar_failed")
            await asyncio.sleep(max(0.0, min(poll_interval, deadline - loop.time())))

    # --- plumbing ------------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        try:
            self._session = utils.http_context.http_session()
        except RuntimeError:
            # Outside a LiveKit job (a script creating avatars, tests): own one.
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def aclose(self) -> None:
        """Close the HTTP session if this client created it."""
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._owns_session = False

    async def __aenter__(self) -> AtmeeAPI:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        data: aiohttp.FormData | None = None,
        params: dict[str, str] | None = None,
        retry: bool = True,
        total_timeout: float = _DEFAULT_TOTAL_TIMEOUT,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """One API call with the plugin's retry policy: transport errors and
        5xx answers are retried up to ``conn_options.max_retry`` times after the
        first attempt; a 503
        ``no_capacity`` and every 4xx are final. ``retry=False`` for calls
        that are not idempotent (creating a session or an avatar).

        ``deadline`` (event-loop time) bounds the whole call, retries and the
        pauses between them included."""
        loop = asyncio.get_running_loop()
        attempts = self._conn_options.max_retry + 1 if retry else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_timeout = total_timeout
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    # the budget ran out: report a timeout, keep the last failure as its cause
                    raise AtmeeException(f"{method} {path} timed out") from last_error
                attempt_timeout = min(total_timeout, remaining)
            try:
                async with self._ensure_session().request(
                    method,
                    f"{self._api_url}{path}",
                    headers={"X-Api-Key": self._api_key, "Accept": "application/json"},
                    json=json,
                    data=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(
                        sock_connect=self._conn_options.timeout, total=attempt_timeout
                    ),
                ) as response:
                    if response.ok:
                        if response.status == 204 or response.content_length == 0:
                            return {}
                        try:
                            payload = await response.json(content_type=None)
                        except ValueError as e:
                            raise AtmeeException(
                                "the Atmee API returned an invalid JSON response",
                                status_code=response.status,
                                code="invalid_response",
                            ) from e
                        return payload if isinstance(payload, dict) else {"data": payload}
                    raise await _error_from_response(response)
            except AtmeeException as e:
                if e.status_code >= 500 and not isinstance(e, AtmeeNoCapacityError):
                    last_error = e
                else:
                    raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = AtmeeException(f"could not reach the Atmee API: {e}")
            if attempt < attempts - 1:
                logger.warning(
                    "atmee api call failed; retrying",
                    extra={"path": path, "attempt": attempt + 1, "error": str(last_error)},
                )
                pause = self._conn_options.retry_interval
                if deadline is not None:
                    pause = min(pause, max(0.0, deadline - loop.time()))
                await asyncio.sleep(pause)
        assert last_error is not None
        raise last_error


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _check_api_url(url: str) -> None:
    """Refuse a plaintext API base: the API key and the avatar's LiveKit join
    token travel in every request. Plain ``http`` is allowed only for a
    loopback host (local development, tests)."""
    parsed = urlsplit(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS:
        return
    raise AtmeeException(
        f"api_url must be an https:// URL (plain http only for localhost), got {url!r}"
    )


async def _error_from_response(response: aiohttp.ClientResponse) -> AtmeeException:
    text = await response.text()
    code: str | None = None
    message = text.strip() or response.reason or f"HTTP {response.status}"
    try:
        body = await response.json(content_type=None)
    except Exception:
        body = None
    if isinstance(body, dict):
        code = body.get("error") if isinstance(body.get("error"), str) else None
        message = str(body.get("message") or body.get("detail") or message)
    if response.status == 503 and code == "no_capacity":
        retry_after: float | None = None
        header = response.headers.get("Retry-After")
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = None
        return AtmeeNoCapacityError(message, retry_after=retry_after)
    if response.status == 409 and code == "avatar_not_renderable":
        return AtmeeAvatarNotReadyError(message, status_code=409, code=code)
    return AtmeeException(message, status_code=response.status, code=code)


def _avatar_info(data: dict[str, Any], *, version: AvatarVersion = "v1") -> AvatarInfo:
    return AvatarInfo(
        avatar_id=str(data.get("avatarId") or data.get("id") or ""),
        status=str(data.get("status", "")),
        kind=data.get("kind"),
        name=data.get("name"),
        version=version,
        raw=data,
    )


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _load_image_file(image: str | Path, content_type: str | None) -> tuple[bytes, str, str]:
    path = Path(image)
    ctype = (
        content_type
        or _IMAGE_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )
    return path.read_bytes(), path.name, ctype


def _extension_for(content_type: str) -> str:
    for ext, ctype in _IMAGE_TYPES.items():
        if ctype == content_type and ext != ".jpeg":
            return ext
    return mimetypes.guess_extension(content_type) or ""
