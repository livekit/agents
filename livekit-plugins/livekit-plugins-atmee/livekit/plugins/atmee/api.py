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

Two resources matter here: **avatars** (read an avatar's generation and
status) and **avatar sessions** (render an avatar,
v1 or v2, into your LiveKit room for your agent, end it). Authentication is
your Atmee API key (``sk_atmee_...``) in the ``X-Api-Key`` header; the key
never reaches a browser.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Literal, get_args
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    utils,
)
from livekit.agents.utils import is_given

from .log import logger

DEFAULT_API_URL = "https://api.atmanity.us"

WaitFor = Literal["initializing", "avatar_joined"]

# Overall budget of the start call, by the avatar's generation. A v1 start is
# acknowledged within seconds and its avatar joins within a couple of minutes.
# The first v2 session after a quiet period can take minutes to start: the API
# asks clients to allow 7 for the avatar to join, and answers `avatar_warming`
# to be asked again meanwhile. An avatar whose generation could not be read
# gets the longer budget.
_START_TOTAL_TIMEOUT: dict[str, dict[str, float]] = {
    "v1": {"initializing": 60.0, "avatar_joined": 180.0},
    "v2": {"initializing": 300.0, "avatar_joined": 450.0},
}
# How long to wait before asking again after `avatar_warming` (or an
# interrupted start) when the server gives no Retry-After, and the bounds on
# what it gives.
_WARMING_RETRY_DEFAULT = 5.0
_WARMING_RETRY_BOUNDS = (1.0, 30.0)
# Engines whose avatars render a selected take (`v2pro` is V2 HD's former name).
_V2_ENGINES = frozenset({"v2", "v2hd", "v2pro"})


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


class AtmeeAvatarWarmingError(AtmeeException):
    """The render could not start yet, and nothing was billed (HTTP 503):
    ``avatar_warming``, a v2 avatar is still getting ready to render, or
    ``avatar_start_interrupted``, the render ended before the avatar joined.

    :meth:`AtmeeAPI.create_avatar_session` asks again by itself within its
    start budget; this is raised only once that budget is spent.
    ``retry_after`` is the server's hint in seconds, when it sent one.
    """

    def __init__(self, message: str, *, code: str, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=503, code=code)
        self.retry_after = retry_after


_RETRYABLE_START_CODES = frozenset({"avatar_warming", "avatar_start_interrupted"})


class AtmeeAvatarNotReadyError(AtmeeException):
    """The avatar cannot be rendered yet (HTTP 409 ``avatar_not_renderable``):
    it has no portrait, or its appearance is still being processed."""


def _warn_avatar_version(avatar_version: NotGivenOr[str]) -> None:
    # Accepted for code written against earlier releases; the avatar decides.
    if is_given(avatar_version):
        logger.warning(
            "avatar_version is deprecated and ignored, the avatar's own engine decides how it "
            "renders"
        )


def _check_wait_for(wait_for: str) -> None:
    allowed = get_args(WaitFor)
    if wait_for not in allowed:
        raise ValueError(
            f"wait_for must be one of {', '.join(repr(v) for v in allowed)}, got {wait_for!r}"
        )


@dataclass
class AvatarSessionInfo:
    """The ``202`` body of ``POST /v1/avatars/{avatarId}/avatar_sessions``."""

    session_id: str
    status: str
    avatar_participant_identity: str
    agent_identity: str
    room_name: str
    max_duration_seconds: int
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AvatarInfo:
    """What ``GET /v1/avatars/{avatarId}`` reports about an avatar.

    ``kind`` (``conversational`` | ``render_only``) says whether the avatar has
    a voice and persona; ``engine`` is its generation as the API reports it
    (``"v1"``, ``"v2"`` or ``"v2hd"``; ``None`` when the API names none).
    """

    avatar_id: str
    status: str
    kind: str | None = None
    name: str | None = None
    engine: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        """Whether it can be rendered: built, and for a v2 avatar, serving a take."""
        if self.status != "ready":
            return False
        return self.engine not in _V2_ENGINES or bool(self.raw.get("servingCandidateId"))


class AtmeeAPI:
    """Async client for the Atmee API.

    Args:
        api_key: your ``sk_atmee_...`` key; defaults to ``ATMEE_API_KEY``.
        api_url: API base; defaults to ``ATMEE_API_URL`` or ``https://api.atmanity.us``.
        conn_options: retry/timeout policy. ``timeout`` bounds the connect of
            every request and each attempt of the calls that are safe to
            repeat (reading and ending a session, reading an avatar), which
            are retried ``max_retry`` times on transport failures and 5xx
            answers. Starting a session is never repeated and has a budget
            of its own.
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
        avatar_version: NotGivenOr[str] = NOT_GIVEN,
        stop: asyncio.Event | None = None,
    ) -> AvatarSessionInfo:
        """Render ``avatar_id`` into the room ``livekit_token`` grants.

        ``livekit_token`` is a LiveKit access token you minted with YOUR
        project's secret for the avatar participant: ``kind=agent``, a
        ``roomJoin`` grant for your room, and the ``lk.publish_on_behalf``
        attribute naming your agent (:class:`AvatarSession` does this for
        you). Returns as soon as the rendering worker acknowledged the start,
        or once the avatar joined with ``wait_for="avatar_joined"``.

        Any avatar renders as what it is, v1 or v2; the request names no
        generation. The call first reads the avatar's generation to know how
        long the start may take: a v1 start may take up to 1 minute (3 with
        ``wait_for="avatar_joined"``). The first v2 session after a quiet
        period can take minutes, so a v2 start may take up to 5 (7.5), and
        while the API answers ``avatar_warming`` or ``avatar_start_interrupted``
        (nothing billed) it asks again after the ``Retry-After`` given.

        Once ``stop`` is set, the call asks no more: it raises an
        ``AtmeeException`` with ``code="closed"`` instead of asking again. A
        request already sent is still awaited, since it may open a session
        that must then be ended.
        """
        _check_wait_for(wait_for)
        _warn_avatar_version(avatar_version)
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
        # Never retried on a timeout or a 5xx: the POST reserves and bills a
        # session as soon as the server accepts it, so asking again could open
        # a second render for the same room. Only the answers that say nothing
        # was reserved (`avatar_warming`, `avatar_start_interrupted`) are asked
        # again, within the start budget.
        loop = asyncio.get_running_loop()
        budget = await self._start_budget(avatar_id, wait_for)
        deadline = loop.time() + budget
        while True:
            if stop is not None and stop.is_set():
                raise AtmeeException("the avatar session start was stopped", code="closed")
            try:
                data = await self._request(
                    "POST",
                    f"/v1/avatars/{avatar_id}/avatar_sessions",
                    json=payload,
                    params=params,
                    retry=False,
                    total_timeout=max(1.0, deadline - loop.time()),
                )
                break
            except AtmeeAvatarWarmingError as e:
                pause = e.retry_after if e.retry_after is not None else _WARMING_RETRY_DEFAULT
                pause = min(max(pause, _WARMING_RETRY_BOUNDS[0]), _WARMING_RETRY_BOUNDS[1])
                if loop.time() + pause >= deadline:
                    raise
                logger.info(
                    "atmee avatar render not started yet; asking again",
                    extra={"code": e.code, "retry_in": pause},
                )
                # nothing is reserved while waiting: a stop ends the wait at once
                if stop is None:
                    await asyncio.sleep(pause)
                else:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=pause)
                    except asyncio.TimeoutError:
                        pass
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
            raw=data,
        )

    async def _start_budget(self, avatar_id: str, wait_for: WaitFor) -> float:
        """How long a start of ``avatar_id`` may take, from its generation."""
        try:
            info = await self._get_avatar(
                avatar_id, deadline=asyncio.get_running_loop().time() + self._conn_options.timeout
            )
        except AtmeeException as e:
            # A key without read access to the avatar, or a slow read: allow
            # for the slower generation; the start reports real failures.
            logger.debug(
                "could not read the atmee avatar's generation; allowing for a v2 start",
                extra={"avatar_id": avatar_id, "error": str(e)},
            )
            return _START_TOTAL_TIMEOUT["v2"][wait_for]
        generation = "v2" if info.engine in _V2_ENGINES else "v1"
        return _START_TOTAL_TIMEOUT[generation][wait_for]

    async def end_avatar_session(self, session_id: str) -> dict[str, Any]:
        """Stop billing for a session now. Idempotent: ending an ended session
        answers normally with ``alreadyEnded: true``. Does not remove the avatar
        from your room — :meth:`AvatarSession.aclose` does that."""
        return await self._request("POST", f"/v1/avatar_sessions/{session_id}/end")

    async def get_avatar_session(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/avatar_sessions/{session_id}")

    # --- avatars -----------------------------------------------------------

    async def get_avatar(self, avatar_id: str) -> AvatarInfo:
        return await self._get_avatar(avatar_id)

    async def _get_avatar(self, avatar_id: str, *, deadline: float | None = None) -> AvatarInfo:
        return _avatar_info(
            await self._request("GET", f"/v1/avatars/{avatar_id}", deadline=deadline)
        )

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
        params: dict[str, str] | None = None,
        retry: bool = True,
        total_timeout: float | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """One API call with the plugin's retry policy: transport errors and
        5xx answers are retried up to ``conn_options.max_retry`` times after the
        first attempt; a 503 ``no_capacity`` or ``avatar_warming`` and every
        4xx are final. ``retry=False`` for calls that are not idempotent
        (creating a session).

        Each attempt is bounded by ``total_timeout``, ``conn_options.timeout``
        when none is given; ``deadline`` (event-loop time) bounds the whole
        call, retries and the pauses between them included."""
        loop = asyncio.get_running_loop()
        attempts = self._conn_options.max_retry + 1 if retry else 1
        per_attempt = self._conn_options.timeout if total_timeout is None else total_timeout
        last_error: Exception | None = None
        for attempt in range(attempts):
            attempt_timeout = per_attempt
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    # the budget ran out: report a timeout, keep the last failure as its cause
                    raise AtmeeException(f"{method} {path} timed out") from last_error
                attempt_timeout = min(per_attempt, remaining)
            try:
                async with self._ensure_session().request(
                    method,
                    f"{self._api_url}{path}",
                    headers={"X-Api-Key": self._api_key, "Accept": "application/json"},
                    json=json,
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
                if e.status_code >= 500 and not isinstance(
                    e, AtmeeNoCapacityError | AtmeeAvatarWarmingError
                ):
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


def _retry_after(response: aiohttp.ClientResponse) -> float | None:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


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
        return AtmeeNoCapacityError(message, retry_after=_retry_after(response))
    if response.status == 503 and code in _RETRYABLE_START_CODES:
        return AtmeeAvatarWarmingError(message, code=code, retry_after=_retry_after(response))
    if response.status == 409 and code == "avatar_not_renderable":
        return AtmeeAvatarNotReadyError(message, status_code=409, code=code)
    return AtmeeException(message, status_code=response.status, code=code)


def _avatar_info(data: dict[str, Any]) -> AvatarInfo:
    engine = data.get("engine")
    return AvatarInfo(
        avatar_id=str(data.get("avatarId") or data.get("id") or ""),
        status=str(data.get("status", "")),
        kind=data.get("kind"),
        name=data.get("name"),
        engine=engine if isinstance(engine, str) and engine else None,
        raw=data,
    )
