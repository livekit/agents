from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions
from livekit.plugins.bey import AvatarSession as BeySession
from livekit.plugins.bithuman import AvatarSession as BitHumanSession
from livekit.plugins.liveavatar.api import LiveAvatarAPI
from livekit.plugins.runway import AvatarSession as RunwaySession

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider", ["bey", "bithuman", "liveavatar", "runway"])
async def test_zero_retries_still_creates_avatar(provider):
    requests = []

    async def create(request):
        requests.append(await request.json())
        if provider == "liveavatar":
            return web.json_response({"data": {"session_id": "session", "session_token": "token"}})
        return web.json_response({"id": "session"})

    app = web.Application()
    app.router.add_post("/{path:.*}", create)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        options = APIConnectOptions(max_retry=0)
        url = str(server.make_url("")).rstrip("/")
        if provider == "bey":
            avatar = BeySession(api_key="key", api_url=url, conn_options=options)
            avatar._http_session = http
            await avatar._start_agent("wss://livekit.test", "token")
        elif provider == "bithuman":
            avatar = BitHumanSession(
                api_secret="secret", avatar_id="avatar", api_url=url, conn_options=options
            )
            avatar._http_session = http
            await avatar._send_request_with_retry({}, json_data={"avatar_id": "avatar"})
        elif provider == "runway":
            avatar = RunwaySession(
                api_key="key", avatar_id="avatar", api_url=url, conn_options=options
            )
            avatar._http_session = http
            avatar._local_participant_identity = "agent"
            await avatar._create_session("wss://livekit.test", "token", "room")
        else:
            api = LiveAvatarAPI(api_key="key", api_url=url, conn_options=options, session=http)
            await api.create_streaming_session(
                livekit_url="wss://livekit.test",
                livekit_token="token",
                room=SimpleNamespace(name="room"),
                avatar_id="avatar",
            )

    assert len(requests) == 1
