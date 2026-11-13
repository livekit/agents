"""Protoface avatar plugin: session termination on close, against a local peer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import http
from livekit.plugins.protoface import AvatarSession

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost


async def test_close_ends_the_billable_session_once(avatar_host: AvatarHost) -> None:
    """Closing the avatar, even twice, ends its Protoface session once.

    https://github.com/livekit/agents/commit/6d33b2050c
    """
    ends: list[str] = []

    async def create(_: web.Request) -> web.Response:
        return web.json_response({"id": "billable-session"})

    async def end(request: web.Request) -> web.Response:
        ends.append(request.match_info["id"])
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/v1/sessions", create)
    app.router.add_post("/v1/sessions/{id}/end", end)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(api_key="key", api_url=str(server.make_url("")))
        await avatar.start(avatar_host.agent, avatar_host.room)

        started = avatar.session_id
        await avatar.aclose()
        await avatar.aclose()

    assert started == "billable-session"
    assert avatar.session_id is None
    assert ends == ["billable-session"]
