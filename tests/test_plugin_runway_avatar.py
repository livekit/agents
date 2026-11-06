"""Tests for the Runway avatar plugin: the retry budget of the session request.

`APIConnectOptions.max_retry` counts retries, not requests, so `max_retry=0` still
owes one initial POST — the contract the core avatar gateway, Anam, Tavus and
Synthesia keep (#7604).
"""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.plugins.runway import AvatarSession

pytestmark = pytest.mark.unit

_ROOM = "room-name"


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status
        self.ok = status < 400

    async def text(self) -> str:
        return f"status {self.status}"

    async def json(self) -> dict[str, Any]:
        return {"id": "realtime-session-1"}

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedSession:
    """Answers each POST with the next step in `script`; an exception instance in the
    script is raised instead, and the last step repeats."""

    def __init__(self, script: list[int | BaseException]) -> None:
        self._script = list(script)
        self.posts = 0

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts += 1
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return _Response(step)


def _avatar(session: _ScriptedSession, max_retry: int) -> AvatarSession:
    avatar = AvatarSession(
        preset_id="preset-id",
        api_key="test-key",
        conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0),
    )
    avatar._http_session = session  # type: ignore[assignment]
    avatar._local_participant_identity = "local-agent"
    return avatar


async def test_zero_retries_still_sends_the_initial_request():
    """`range(max_retry)` with max_retry=0 looped zero times, so startup failed
    without ever asking the API — retries were disabled along with the request."""
    session = _ScriptedSession([200])

    await _avatar(session, max_retry=0)._create_session("wss://livekit.test", "token", _ROOM)

    assert session.posts == 1


@pytest.mark.parametrize("max_retry", [0, 1, 3])
async def test_a_persistent_connection_error_is_retried_max_retry_times(
    max_retry: int,
):
    """One initial request plus `max_retry` retries, and the last error is kept."""
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused")])

    with pytest.raises(APIConnectionError):
        await _avatar(session, max_retry=max_retry)._create_session(
            "wss://livekit.test", "token", _ROOM
        )

    assert session.posts == max_retry + 1


async def test_a_retryable_error_followed_by_success_starts_the_session():
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused"), 200])

    await _avatar(session, max_retry=1)._create_session("wss://livekit.test", "token", _ROOM)

    assert session.posts == 2


async def test_a_client_error_is_not_retried():
    """A bad key or an unknown preset fails the same way every time."""
    session = _ScriptedSession([401])

    with pytest.raises(APIStatusError) as exc_info:
        await _avatar(session, max_retry=3)._create_session("wss://livekit.test", "token", _ROOM)

    assert session.posts == 1
    assert exc_info.value.status_code == 401
