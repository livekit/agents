"""Beyond Presence avatar plugin: session creation through a local peer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import api
from livekit.agents import APIConnectOptions, APIStatusError, http
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.plugins.bey import AvatarSession

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost

_OPTS = APIConnectOptions(max_retry=2, retry_interval=0, connect_timeout=1, read_timeout=1)


@pytest.fixture
async def peer(
    avatar_host: AvatarHost,
) -> AsyncIterator[tuple[AvatarSession, list[int], list[dict[str, str]]]]:
    """A local Beyond Presence API answering with the queued statuses; yields avatar and state."""
    statuses: list[int] = []
    requests: list[dict[str, str]] = []

    async def create(request: web.Request) -> web.Response:
        requests.append(await request.json())
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        return web.json_response({"detail": f"provider says {status}"}, status=status)

    app = web.Application()
    app.router.add_post("/v1/session", create)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(
            api_key="bey-key", api_url=str(server.make_url("")).rstrip("/"), conn_options=_OPTS
        )
        yield avatar, statuses, requests


async def test_explicit_livekit_credentials_sign_the_avatar_token(
    peer: tuple[AvatarSession, list[int], list[dict[str, str]]],
    avatar_host: AvatarHost,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The avatar joins with the credentials passed to start(), not the environment's.

    https://github.com/livekit/agents/pull/2217
    """
    avatar, statuses, requests = peer
    statuses.append(200)
    monkeypatch.setenv("LIVEKIT_URL", "wss://env.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "env-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "e" * 32)

    await avatar.start(
        avatar_host.agent,
        avatar_host.room,
        livekit_url="wss://given.test",
        livekit_api_key="given-key",
        livekit_api_secret="g" * 32,
    )

    [sent] = requests
    assert sent["livekit_url"] == "wss://given.test"
    claims = api.TokenVerifier("given-key", "g" * 32).verify(sent["livekit_token"])
    assert claims.video is not None and claims.video.room == "room-1"
    assert claims.attributes == {ATTRIBUTE_PUBLISH_ON_BEHALF: "agent-1"}


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/7495 reproduces: "
    "every status is retried and replaced by a generic APIConnectionError",
)
@pytest.mark.parametrize(("status", "attempts"), [(401, 1), (503, _OPTS.max_retry + 1)])
async def test_provider_status_survives_and_only_transient_errors_retry(
    peer: tuple[AvatarSession, list[int], list[dict[str, str]]],
    avatar_host: AvatarHost,
    status: int,
    attempts: int,
) -> None:
    """A refused session keeps the provider's status and body; only 5xx is retried.

    https://github.com/livekit/agents/issues/7495
    """
    avatar, statuses, requests = peer
    statuses.append(status)

    with pytest.raises(APIStatusError) as error:
        await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    assert error.value.status_code == status
    assert f"provider says {status}" in str(error.value.body)
    assert len(requests) == attempts
