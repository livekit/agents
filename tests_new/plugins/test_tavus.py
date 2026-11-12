"""Tavus avatar: HTTP retry and error contracts of session start, against a local peer.

https://github.com/livekit/agents/issues/7493
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, http
from livekit.plugins.tavus import AvatarSession

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost

OPTIONS = APIConnectOptions(max_retry=2, retry_interval=0, connect_timeout=1, read_timeout=1)


@dataclass
class Tavus:
    """The local Tavus API: answers each conversation request with the next status (the last
    repeats; ``"disconnect"`` drops the connection) and counts the requests."""

    script: list[int | str] = field(default_factory=list)
    calls: int = 0
    api_url: str = ""


@pytest.fixture
async def tavus() -> AsyncIterator[Tavus]:
    peer = Tavus()

    async def respond(request: web.Request) -> web.Response:
        peer.calls += 1
        outcome = peer.script.pop(0) if len(peer.script) > 1 else peer.script[0]
        if outcome == "disconnect":
            assert request.transport is not None
            request.transport.close()
            return web.Response()
        assert isinstance(outcome, int)
        return web.json_response({"conversation_id": "c1"}, status=outcome)

    app = web.Application()
    app.router.add_post("/{path:.*}", respond)
    async with TestServer(app) as server, http.open():
        peer.api_url = str(server.make_url("/v2")).rstrip("/")
        yield peer


async def _start(tavus: Tavus, avatar_host: AvatarHost) -> AvatarSession:
    avatar = AvatarSession(
        face_id="face", api_key="key", api_url=tavus.api_url, conn_options=OPTIONS
    )
    await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
    return avatar


async def test_client_error_is_not_retried_and_keeps_its_status(
    tavus: Tavus, avatar_host: AvatarHost
) -> None:
    """A 4xx fails start at once with its status."""
    tavus.script.append(401)

    with pytest.raises(APIStatusError) as error:
        await _start(tavus, avatar_host)

    assert error.value.status_code == 401
    assert tavus.calls == 1


async def test_server_error_is_retried_until_success(tavus: Tavus, avatar_host: AvatarHost) -> None:
    """A 5xx is retried until the conversation is created."""
    tavus.script.extend([503, 503, 200])

    avatar = await _start(tavus, avatar_host)

    assert avatar.conversation_id == "c1"
    assert tavus.calls == 3


async def test_persistent_server_error_is_raised_after_max_retry_retries(
    tavus: Tavus, avatar_host: AvatarHost
) -> None:
    """A 5xx that persists fails start with its status after max_retry retries."""
    tavus.script.append(503)

    with pytest.raises(APIStatusError) as error:
        await _start(tavus, avatar_host)

    assert error.value.status_code == 503
    assert tavus.calls == OPTIONS.max_retry + 1


async def test_network_error_is_retried_and_chained(tavus: Tavus, avatar_host: AvatarHost) -> None:
    """A dropped connection is retried, then fails with the transport error as its cause."""
    tavus.script.append("disconnect")

    with pytest.raises(APIConnectionError) as error:
        await _start(tavus, avatar_host)

    assert isinstance(error.value.__cause__, APIConnectionError)
    assert tavus.calls == OPTIONS.max_retry + 1
