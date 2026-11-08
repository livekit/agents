from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.avatario.api import AvatarioAPI
from livekit.plugins.bey import AvatarSession as BeySession
from livekit.plugins.bithuman import AvatarSession as BitHumanSession
from livekit.plugins.liveavatar.api import LiveAvatarAPI
from livekit.plugins.runway import AvatarSession as RunwaySession

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provider", ["avatario", "bey", "bithuman", "liveavatar", "runway"])
async def test_zero_retries_still_creates_avatar(provider, monkeypatch):
    requests = []

    async def create(request):
        requests.append(await request.json())
        if provider == "liveavatar":
            return web.json_response({"data": {"session_id": "session", "session_token": "token"}})
        return web.json_response({"id": "session"})

    app = web.Application()
    app.router.add_post("/{path:.*}", create)
    async with TestServer(app) as server, http.open() as session:
        options = APIConnectOptions(max_retry=0)
        url = str(server.make_url("")).rstrip("/")
        if provider == "avatario":
            monkeypatch.setattr("livekit.plugins.avatario.api.DEFAULT_API_URL", url)
            api = AvatarioAPI(
                api_key="key", avatar_id="avatar", session=session, conn_options=options
            )
            await api.start_session(livekit_agent_identity="agent")
        elif provider == "bey":
            avatar = BeySession(api_key="key", api_url=url, conn_options=options)
            avatar._http_session = session
            await avatar._start_agent("wss://livekit.test", "token")
        elif provider == "bithuman":
            avatar = BitHumanSession(
                api_secret="secret", avatar_id="avatar", api_url=url, conn_options=options
            )
            avatar._http_session = session
            await avatar._send_request_with_retry({}, json_data={"avatar_id": "avatar"})
        elif provider == "runway":
            avatar = RunwaySession(
                api_key="key", avatar_id="avatar", api_url=url, conn_options=options
            )
            avatar._http_session = session
            avatar._local_participant_identity = "agent"
            await avatar._create_session("wss://livekit.test", "token", "room")
        else:
            api = LiveAvatarAPI(api_key="key", api_url=url, conn_options=options, session=session)
            await api.create_streaming_session(
                livekit_url="wss://livekit.test",
                livekit_token="token",
                room=SimpleNamespace(name="room"),
                avatar_id="avatar",
            )

    assert len(requests) == 1
