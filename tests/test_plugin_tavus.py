"""Tests for the Tavus plugin: retry classification when calling the Tavus API.

Same behaviour as the Anam fix in #7314: a 4xx fails on the first attempt with the
provider's status, a 5xx is retried `max_retry` times, and a 5xx that persists is
raised as the provider's error rather than a generic connection error.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.plugin("tavus")

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=1.0)


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status
        self.ok = status < 400

    async def text(self) -> str:
        return f"status {self.status}"

    async def json(self) -> dict[str, Any]:
        return {"conversation_id": "c1"}

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _ScriptedSession:
    """Answers each POST with the next status in `script`; an exception instance in
    the script is raised instead."""

    def __init__(self, script: list[int | BaseException]) -> None:
        self._script = list(script)
        self.posts = 0

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts += 1
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return _Response(step)


def _api(session: _ScriptedSession):
    from livekit.plugins.tavus.api import TavusAPI

    return TavusAPI(
        api_key="test-key",
        api_url="http://tavus.test",
        conn_options=_OPTS,
        session=session,  # type: ignore[arg-type]
    )


async def test_client_error_is_not_retried_and_keeps_its_status():
    """A bad API key fails the same way every time; retrying it only delays the
    error and replaced the 401 with a generic 'after all retries' message."""
    session = _ScriptedSession([401])

    with pytest.raises(APIStatusError) as exc_info:
        await _api(session)._post("conversations", {})

    assert session.posts == 1
    assert exc_info.value.status_code == 401


async def test_server_error_is_retried_until_success():
    session = _ScriptedSession([503, 503, 200])

    data = await _api(session)._post("conversations", {})

    assert data == {"conversation_id": "c1"}
    assert session.posts == 3


async def test_persistent_server_error_is_raised_after_max_retry_retries():
    """One initial attempt plus `max_retry` retries, like Anam, and the final error
    is the provider's status rather than a generic connection error."""
    session = _ScriptedSession([503])

    with pytest.raises(APIStatusError) as exc_info:
        await _api(session)._post("conversations", {})

    assert session.posts == _OPTS.max_retry + 1
    assert exc_info.value.status_code == 503


async def test_malformed_success_body_is_not_retried():
    """Tavus already accepted the POST, so retrying could create a duplicate
    conversation; fail once with a non-retryable error chained to the decode error."""

    class _BadJSONResponse(_Response):
        async def json(self) -> dict[str, Any]:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class _BadJSONSession(_ScriptedSession):
        def post(self, url: str, **kwargs: Any) -> _Response:
            self.posts += 1
            return _BadJSONResponse(200)

    session = _BadJSONSession([200])

    with pytest.raises(APIConnectionError) as exc_info:
        await _api(session)._post("conversations", {})

    assert session.posts == 1
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_network_error_is_retried_and_chained():
    session = _ScriptedSession([aiohttp.ClientConnectionError("connection refused")])

    with pytest.raises(APIConnectionError) as exc_info:
        await _api(session)._post("conversations", {})

    assert session.posts == _OPTS.max_retry + 1
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectionError)
