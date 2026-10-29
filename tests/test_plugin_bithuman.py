"""Tests for the bitHuman plugin: retry classification when starting a cloud session.

Same behaviour as the Anam fix in #7314: a 4xx fails on the first attempt with the
provider's status, a 5xx is retried `max_retry` times, and a 5xx that persists is
raised as the provider's error rather than a generic connection error.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("bithuman")

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status
        self.ok = status < 400

    async def text(self) -> str:
        return f"status {self.status}"

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedSession:
    """Answers each POST with the next status in `script`; an exception instance in
    the script is raised instead. The last entry repeats."""

    def __init__(self, script: list[int | BaseException]) -> None:
        self._script = list(script)
        self.posts = 0

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts += 1
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return _Response(step)


def _avatar(session: _ScriptedSession):
    from livekit.plugins.bithuman import AvatarSession

    avatar = AvatarSession(
        api_url="http://bithuman.test",
        api_secret="test-secret",
        avatar_id="avatar-1",
        conn_options=_OPTS,
    )
    avatar._http_session = session  # type: ignore[assignment]
    return avatar


async def _send(avatar) -> None:
    await avatar._send_request_with_retry(headers={}, json_data={"agent_id": "avatar-1"})


async def test_client_error_is_not_retried_and_keeps_its_status():
    """A bad API secret fails the same way every time; retrying it only delays the
    error and replaced the 401 with a generic 'after all retries' message."""
    session = _ScriptedSession([401])

    with pytest.raises(APIStatusError) as exc_info:
        await _send(_avatar(session))

    assert session.posts == 1
    assert exc_info.value.status_code == 401


async def test_server_error_is_retried_until_success():
    session = _ScriptedSession([503, 503, 200])

    await _send(_avatar(session))

    assert session.posts == 3


async def test_persistent_server_error_is_raised_after_max_retry_retries():
    """One initial attempt plus `max_retry` retries, like Anam, and the final error
    is the provider's status rather than a generic connection error."""
    session = _ScriptedSession([503])

    with pytest.raises(APIStatusError) as exc_info:
        await _send(_avatar(session))

    assert session.posts == _OPTS.max_retry + 1
    assert exc_info.value.status_code == 503


async def test_network_error_is_retried_and_chained():
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused")])

    with pytest.raises(APIConnectionError) as exc_info:
        await _send(_avatar(session))

    assert session.posts == _OPTS.max_retry + 1
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionError)
