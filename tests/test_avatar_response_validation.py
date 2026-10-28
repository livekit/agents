"""A successful avatar POST with invalid JSON must never be repeated."""

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.anam.api import AnamAPI
from livekit.plugins.anam.types import PersonaConfig
from livekit.plugins.avatario.api import AvatarioAPI
from livekit.plugins.avatartalk.api import AvatarTalkAPI
from livekit.plugins.did.api import DIDAPI
from livekit.plugins.keyframe.api import KeyframeAPI
from livekit.plugins.lemonslice.api import LemonSliceAPI
from livekit.plugins.liveavatar.api import LiveAvatarAPI

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "provider", ["anam", "avatario", "avatartalk", "did", "keyframe", "lemonslice", "liveavatar"]
)
async def test_invalid_success_response_does_not_repeat_creation(provider, monkeypatch):
    requests = []

    async def create(request):
        requests.append(await request.json())
        return web.json_response([])

    app = web.Application()
    app.router.add_post("/{path:.*}", create)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        options = APIConnectOptions(max_retry=2, retry_interval=0)
        url = str(server.make_url("")).rstrip("/")
        if provider == "anam":
            api = AnamAPI(api_key="key", api_url=url, session=http, conn_options=options)
            call = api.start_session(
                PersonaConfig(name="avatar", avatarId="avatar"), "wss://livekit.test", "token"
            )
        elif provider == "did":
            api = DIDAPI(api_key="key", api_url=url, session=http, conn_options=options)
            call = api.join_session(agent_id="avatar", transport={}, audio_config={})
        elif provider == "avatartalk":
            api = AvatarTalkAPI(api_key="key", api_url=url)
            call = api.start_session(
                "wss://livekit.test", "avatar", "neutral", "room", "listener", "token", "agent"
            )
        elif provider == "keyframe":
            api = KeyframeAPI(api_key="key", api_url=url, session=http, conn_options=options)
            call = api.create_plugin_session(
                room_name="room",
                livekit_url="wss://livekit.test",
                livekit_token="token",
                source_participant_identity="agent",
            )
        elif provider == "avatario":
            monkeypatch.setattr("livekit.plugins.avatario.api.DEFAULT_API_URL", url)
            api = AvatarioAPI(api_key="key", avatar_id="avatar", session=http, conn_options=options)
            call = api.start_session(livekit_agent_identity="agent")
        elif provider == "lemonslice":
            api = LemonSliceAPI(api_key="key", api_url=url, session=http, conn_options=options)
            call = api.start_agent_session(
                livekit_url="wss://livekit.test",
                livekit_token="token",
                livekit_session_id="room",
                agent_id="avatar",
            )
        else:
            api = LiveAvatarAPI(api_key="key", api_url=url, session=http, conn_options=options)
            call = api.start_streaming_session("session", "token")
        with pytest.raises(APIConnectionError) as error:
            await call
        assert error.value.retryable is False
    assert len(requests) == 1
