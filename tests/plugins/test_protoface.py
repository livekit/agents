"""Protoface avatar plugin: session termination on close, against a local peer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import http
from livekit.plugins.protoface import AvatarSession

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost


async def test_close_retains_session_until_termination_succeeds(avatar_host: AvatarHost) -> None:
    """A failed end keeps the session so the next close ends it again.

    https://github.com/livekit/agents/commit/6d33b2050c
    """
    ends: list[str] = []

    async def create(_: web.Request) -> web.Response:
        return web.json_response({"id": "billable-session"})

    async def end(request: web.Request) -> web.Response:
        ends.append(request.match_info["id"])
        status = 400 if len(ends) == 1 else 200
        return web.json_response({}, status=status)

    app = web.Application()
    app.router.add_post("/v1/sessions", create)
    app.router.add_post("/v1/sessions/{id}/end", end)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(api_key="key", api_url=str(server.make_url("")))
        await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

        await avatar.aclose()
        after_failure = avatar.session_id
        await avatar.aclose()

    assert after_failure == "billable-session"
    assert avatar.session_id is None
    assert ends == ["billable-session", "billable-session"]
