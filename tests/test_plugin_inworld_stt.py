"""Inworld speech recognition rejects malformed provider frames."""

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins import inworld

pytestmark = pytest.mark.plugin("inworld")


@pytest.mark.parametrize("body", ["{", "[]", '{"result":{"transcription":{"transcript":42}}}'])
async def test_malformed_stream_frame_is_terminal_provider_error(body: str) -> None:
    async def recognize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(body)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/stt/v1/transcribe:streamBidirectional", recognize)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        model = inworld.STT(
            api_key="test", base_url=str(server.make_url("/")), http_session=session
        )
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            with pytest.raises(APIConnectionError) as failure:
                await asyncio.wait_for(anext(stream), 0.5)
            assert not failure.value.retryable
            assert isinstance(failure.value.__cause__, ValidationError)
        await model.aclose()
