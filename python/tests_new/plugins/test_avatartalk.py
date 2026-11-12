"""AvatarTalk avatar plugin: session start failures and close, against a local peer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import http
from livekit.plugins.avatartalk import AvatarSession
from livekit.plugins.avatartalk.api import AvatarTalkException

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost


async def test_start_propagates_provider_failure(avatar_host: AvatarHost) -> None:
    """A rejected session start surfaces and leaves the agent's audio untouched."""

    async def create(_: web.Request) -> web.Response:
        return web.json_response({"detail": "rejected"}, status=400)

    app = web.Application()
    app.router.add_post("/livekit/create-session", create)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(api_secret="key", api_url=str(server.make_url("")).rstrip("/"))

        with pytest.raises(AvatarTalkException, match="rejected"):
            await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    avatar_host.audio_tail.assert_not_called()
    avatar_host.framework_start.assert_not_awaited()


async def test_close_stops_the_billable_session_once(avatar_host: AvatarHost) -> None:
    """Closing the avatar twice deletes the provider session exactly once."""
    deletes: list[str] = []

    async def create(_: web.Request) -> web.Response:
        return web.json_response({"task_id": "billable-session"})

    async def delete(request: web.Request) -> web.Response:
        deletes.append(request.match_info["task_id"])
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/livekit/create-session", create)
    app.router.add_delete("/livekit/delete-session/{task_id}", delete)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(api_secret="key", api_url=str(server.make_url("")).rstrip("/"))
        await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

        await avatar.aclose()
        await avatar.aclose()

    assert deletes == ["billable-session"]
