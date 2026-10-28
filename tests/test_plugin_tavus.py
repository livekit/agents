"""Tavus HTTP retry/error contracts through real local peers."""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.plugins.tavus.api import TavusAPI

pytestmark = pytest.mark.plugin("tavus")
_OPTS = APIConnectOptions(max_retry=2, retry_interval=0, connect_timeout=1, read_timeout=1)


@pytest.fixture
async def peer():
    script = []
    calls = []

    async def respond(request):
        calls.append(await request.json())
        outcome = script.pop(0) if len(script) > 1 else script[0]
        if outcome == "malformed":
            return web.Response(text="not JSON", content_type="application/json")
        if outcome == "disconnect":
            request.transport.close()
            return web.Response()
        return web.json_response({"conversation_id": "c1"}, status=outcome)

    app = web.Application()
    app.router.add_post("/{path:.*}", respond)
    async with TestServer(app) as server, aiohttp.ClientSession() as http:
        api = TavusAPI(
            api_key="key",
            api_url=str(server.make_url("/v2")).rstrip("/"),
            session=http,
            conn_options=_OPTS,
        )
        yield api, script, calls


async def test_client_error_is_not_retried_and_keeps_its_status(peer):
    api, script, calls = peer
    script.append(401)
    with pytest.raises(APIStatusError) as error:
        await api.create_conversation()
    assert error.value.status_code == 401
    assert len(calls) == 1


async def test_server_error_is_retried_until_success(peer):
    api, script, calls = peer
    script.extend([503, 503, 200])
    assert await api.create_conversation() == "c1"
    assert len(calls) == 3


async def test_persistent_server_error_is_raised_after_max_retry_retries(peer):
    api, script, calls = peer
    script.append(503)
    with pytest.raises(APIStatusError) as error:
        await api.create_conversation()
    assert error.value.status_code == 503
    assert len(calls) == _OPTS.max_retry + 1


async def test_malformed_success_body_is_not_retried(peer):
    api, script, calls = peer
    script.append("malformed")
    with pytest.raises(APIConnectionError) as error:
        await api.create_conversation()
    assert error.value.retryable is False
    assert isinstance(error.value.__cause__, ValueError)
    assert len(calls) == 1


async def test_network_error_is_retried_and_chained(peer):
    api, script, calls = peer
    script.append("disconnect")
    with pytest.raises(APIConnectionError) as error:
        await api.create_conversation()
    assert isinstance(error.value.__cause__, aiohttp.ClientError)
    assert len(calls) == _OPTS.max_retry + 1
