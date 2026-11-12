"""Nabrah STT plugin: provider error redaction, against a local peer."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIStatusError, http
from livekit.plugins.nabrah import STT

TRANSCRIPT = "private customer transcript"


async def _listen(request: web.Request) -> web.WebSocketResponse:
    socket = web.WebSocketResponse()
    await socket.prepare(request)
    await socket.receive()  # the configuration frame
    await socket.send_json({"type": "ready"})
    await socket.send_json({"type": "error", "message": TRANSCRIPT})
    async for _ in socket:
        pass
    return socket


async def test_provider_error_does_not_expose_provider_message() -> None:
    """A provider error fails the stream without its message. https://github.com/livekit/agents/pull/7337"""
    app = web.Application()
    app.router.add_get("/listen", _listen)
    async with TestServer(app) as server, http.open():
        provider = STT(api_key="key", base_url=str(server.make_url("/listen").with_scheme("ws")))

        with pytest.raises(APIStatusError) as caught:
            async with provider.stream() as stream:
                async for _ in stream:
                    pass

    assert TRANSCRIPT not in str(caught.value)
