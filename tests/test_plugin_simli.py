"""Tests for the Simli plugin: starting a session.

Covers the retry behaviour (same shape as the Anam fix in #7314) and that a failed
start logs and returns instead of crashing inside its own error handler.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("simli")

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)
_TOKEN_BODY = json.dumps({"session_token": "simli-session-token"})


class _Response:
    """Usable both as `await session.post(...)` and `async with session.post(...)`."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.ok = status < 400
        self._body = body

    async def text(self) -> str:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=self.status
            )

    def __await__(self):
        async def _self() -> _Response:
            return self

        return _self().__await__()

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedSession:
    """Answers each POST with the next step in `script`: a status code (body is the
    token response for 2xx), or an exception instance to raise. The last step repeats."""

    timeout = aiohttp.ClientTimeout(total=300)

    def __init__(self, script: list[int | BaseException]) -> None:
        self._script = list(script)
        self.posts = 0
        self.timeouts: list[aiohttp.ClientTimeout] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts += 1
        if "timeout" in kwargs:
            self.timeouts.append(kwargs["timeout"])
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return _Response(step, _TOKEN_BODY if step < 400 else f"status {step}")


def _avatar(monkeypatch: pytest.MonkeyPatch, session: _ScriptedSession):
    from livekit.agents import utils
    from livekit.plugins.simli import AvatarSession, SimliConfig

    monkeypatch.setattr(utils.http_context, "http_session", lambda: session)
    return AvatarSession(
        simli_config=SimliConfig(api_key="test-key", face_id="face"),
        api_url="http://simli.test",
        conn_options=_OPTS,
    )


async def test_client_error_is_not_retried_and_keeps_its_status(monkeypatch):
    session = _ScriptedSession([401])

    with pytest.raises(APIStatusError) as exc_info:
        await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert session.posts == 1
    assert exc_info.value.status_code == 401


async def test_server_error_is_retried_until_success(monkeypatch):
    session = _ScriptedSession([503, 503, 200])

    body = await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert body == _TOKEN_BODY
    assert session.posts == 3


async def test_request_timeout_keeps_the_session_total_and_uses_connect_option(monkeypatch):
    session = _ScriptedSession([200])

    await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert session.timeouts[0].total == 300
    assert session.timeouts[0].sock_connect == _OPTS.timeout


async def test_network_error_is_retried_and_chained(monkeypatch):
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused")])

    with pytest.raises(APIConnectionError) as exc_info:
        await _avatar(monkeypatch, session)._post_with_retry("http://simli.test/x", payload={})

    assert session.posts == _OPTS.max_retry + 1
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionError)


async def test_failed_connect_request_logs_and_returns(monkeypatch, caplog):
    """If the second request fails before a response exists, the old error handler
    referenced the unassigned response and raised UnboundLocalError. start() should
    log the failure, leave the agent's audio output alone, and return."""
    from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
    from livekit.plugins.simli import avatar as simli_avatar

    async def _base_start(self, agent_session, room):
        return None

    monkeypatch.setattr(BaseAvatarSession, "start", _base_start)
    monkeypatch.setattr(
        simli_avatar,
        "get_job_context",
        lambda: SimpleNamespace(local_participant_identity="agent"),
    )

    session = _ScriptedSession([200, aiohttp.ClientConnectionError("connection reset")])
    avatar = _avatar(monkeypatch, session)
    agent_session = MagicMock()

    with caplog.at_level(logging.ERROR, logger="livekit.plugins.simli"):
        await avatar.start(
            agent_session,
            SimpleNamespace(name="room"),  # type: ignore[arg-type]
            livekit_url="wss://livekit.test",
            livekit_api_key="key",
            livekit_api_secret="secret" * 8,
        )

    agent_session.output.replace_audio_tail.assert_not_called()
    assert any("simli" in r.getMessage() for r in caplog.records)
    assert not any("simli-session-token" in r.getMessage() for r in caplog.records)
