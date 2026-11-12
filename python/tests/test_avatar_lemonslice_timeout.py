import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions
from livekit.plugins.lemonslice.meeting.audio import stream_meeting_relay

pytestmark = pytest.mark.unit


async def test_meeting_relay_recovers_from_stalled_handshake():
    stop = asyncio.Event()
    release = asyncio.Event()
    requests = 0

    async def stalled_handshake(_):
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
