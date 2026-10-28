import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APITimeoutError
from livekit.plugins.liveavatar import AvatarSession

pytestmark = pytest.mark.unit


async def test_websocket_handshake_obeys_read_timeout():
    release = asyncio.Event()

    async def stalled_handshake(_):
        await release.wait()
        return web.Response()

    app = web.Application()
    app.router.add_get("/socket", stalled_handshake)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        options = APIConnectOptions(read_timeout=0.02, max_retry=0)
        avatar = AvatarSession(api_key="key", avatar_id="avatar", conn_options=options)
        avatar._api._session = http
        avatar._ws_url = str(server.make_url("/socket"))
        try:
            with pytest.raises(APITimeoutError) as error:
                await asyncio.wait_for(avatar._main_task(), 0.5)
            assert error.value.phase == "read"
        finally:
            release.set()
