"""Simli avatar plugin: session startup, against a local peer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, http
from livekit.plugins.simli import AvatarSession, SimliConfig

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost


@pytest.mark.parametrize("failed_request", ["/compose/token", "/integrations/livekit/agents"])
async def test_start_propagates_provider_failure(
    avatar_host: AvatarHost, failed_request: str
) -> None:
    """A dropped Simli request fails start before the agent's audio is rerouted.

    https://github.com/livekit/agents/issues/7497
    """

    async def answer(request: web.Request) -> web.Response:
        if request.path == failed_request:
            assert request.transport is not None
            request.transport.close()
        return web.json_response({"session_token": "token"})

    app = web.Application()
    app.router.add_post("/{path:.*}", answer)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(
            simli_config=SimliConfig(api_key="key", face_id="face"),
            api_url=str(server.make_url("")).rstrip("/"),
        )

        with pytest.raises(APIConnectionError):
            await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    avatar_host.audio_tail.assert_not_called()
    avatar_host.framework_start.assert_not_awaited()
