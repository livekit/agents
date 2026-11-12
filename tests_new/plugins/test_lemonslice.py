"""LemonSlice avatar plugin: session startup and meeting relay reconnects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions
from livekit.plugins.lemonslice import AvatarSession
from livekit.plugins.lemonslice.meeting.audio import stream_meeting_relay
from tests_new.fakes import until

if TYPE_CHECKING:
    from tests_new.plugins.conftest import AvatarHost


async def test_audio_output_is_bound_before_the_session_request_answers(
    avatar_host: AvatarHost,
) -> None:
    """Replies started while LemonSlice creates the session already go to the avatar.

    https://github.com/livekit/agents/pull/5837
    """
    bound_at_request: list[bool] = []
    release = asyncio.Event()

    async def create(_: web.Request) -> web.Response:
        replaced = avatar_host.audio_tail.call_args_list
        bound_at_request.append(replaced == [((avatar_host.sinks[0],),)])
        await release.wait()
        return web.json_response({"session_id": "S1"})

    app = web.Application()
    app.router.add_post("/sessions", create)
    async with TestServer(app) as server:
        avatar = AvatarSession(
            agent_id="agent", api_key="key", api_url=str(server.make_url("/sessions"))
        )
        start = asyncio.create_task(
            avatar.start(avatar_host.agent, avatar_host.room, **avatar_host.credentials)
        )
        await until(lambda: bound_at_request, "LemonSlice receives the session request")
        pending_while_requesting = not start.done()
        release.set()

        assert await start == "S1"

    assert bound_at_request == [True]
    assert pending_while_requesting


async def test_meeting_relay_recovers_from_stalled_handshake() -> None:
    """Fix 2c12a14a61: relay sockets inherit the read timeout and reconnect after a stall."""
    stop = asyncio.Event()
    release = asyncio.Event()
    requests = 0

    async def stalled_handshake(_: web.Request) -> web.StreamResponse:
        nonlocal requests
        requests += 1
        if requests == 2:
            stop.set()
            release.set()
        await release.wait()
        return web.Response()

    app = web.Application()
    app.router.add_get("/socket", stalled_handshake)
    async with TestServer(app) as server:
        try:
            await asyncio.wait_for(
                stream_meeting_relay(
                    str(server.make_url("/socket")),
                    lambda _: None,
                    stop=stop,
                    reconnect_delay_s=0,
                    conn_options=APIConnectOptions(read_timeout=0.02, max_retry=0),
                ),
                0.5,
            )
            assert requests == 2
        finally:
            release.set()
