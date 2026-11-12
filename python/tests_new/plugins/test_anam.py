"""Anam avatar plugin: the agent's audio sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import http
from livekit.plugins.anam import AvatarSession, PersonaConfig

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8042 reproduces: the Anam sink does not "
    "wait for lk.playback_started, so a reply starts when its first frame is sent",
)
async def test_reply_starts_when_the_avatar_reports_playback(avatar_host: AvatarHost) -> None:
    """The agent's reply starts when Anam reports lk.playback_started, not at its first frame.

    https://github.com/livekit/agents/issues/3576 https://github.com/livekit/agents/pull/8042
    """

    async def create(_: web.Request) -> web.Response:
        return web.json_response({"sessionId": "S1"})

    app = web.Application()
    app.router.add_post("/v1/engine/session", create)
    async with TestServer(app) as server, http.open():
        avatar = AvatarSession(
            persona_config=PersonaConfig(name="n", avatarId="a"),
            api_key="key",
            api_url=str(server.make_url("")).rstrip("/"),
        )

        await avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)

    assert avatar.session_id == "S1"
    [sink] = avatar_host.sinks
    options = sink.options
    assert options["destination_identity"] == avatar.avatar_identity
    assert options["wait_remote_track"] == rtc.TrackKind.KIND_VIDEO
    assert options.get("wait_playback_start") is True
