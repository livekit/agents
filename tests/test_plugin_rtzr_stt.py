import asyncio
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions
from livekit.plugins.rtzr import stt as rtzr_stt
from livekit.plugins.rtzr.rtzrapi import RTZROpenAPIClient

pytestmark = pytest.mark.unit


@pytest.fixture
async def gateway(monkeypatch):
    pongs = asyncio.Queue()

    async def handler(request):
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        await ws.ping(b"probe")
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.PONG:
                pongs.put_nowait(msg.data)
                await ws.send_json({})
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/transcribe:streaming", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        client = RTZROpenAPIClient(client_id="test", client_secret="test", http_session=session)
        client._ws_base = str(server.make_url("")).rstrip("/").replace("http://", "ws://")
        monkeypatch.setattr(client, "get_token", AsyncMock(return_value="test-token"))
        yield client, pongs


async def test_standalone_client_answers_server_ping(gateway):
    client, pongs = gateway
    ws = await client.connect_websocket({})
    try:
        msg = await asyncio.wait_for(ws.receive(), 1)
        assert msg.type == aiohttp.WSMsgType.TEXT
        assert pongs.get_nowait() == b"probe"
    finally:
        await ws.close()


async def test_stt_monitor_answers_server_ping(gateway, monkeypatch):
    client, pongs = gateway
    connect = AsyncMock(wraps=client.connect_websocket)
    monkeypatch.setattr(client, "connect_websocket", connect)
    monkeypatch.setattr(rtzr_stt, "RTZROpenAPIClient", lambda **kwargs: client)
    provider = rtzr_stt.STT()
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    ws = None
    try:
        ws = await stream._connect_ws()
        await asyncio.wait_for(stream._recv_loop(ws), 1)
        assert pongs.get_nowait() == b"probe"
        assert connect.call_args.kwargs["autoping"] is False
    finally:
        if ws is not None:
            await ws.close()
        await stream.aclose()
        await provider.aclose()
