"""Gradium plugin: streaming STT setup message, against a local peer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import http
from livekit.plugins.gradium import STT

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


async def test_speech_stream_sends_language_in_json_config() -> None:
    """The setup message carries the language and temperature. https://github.com/livekit/agents/pull/5878"""
    setups: asyncio.Queue[Wire] = asyncio.Queue()

    async def asr(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        setups.put_nowait(json.loads((await socket.receive()).data))
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/asr", asr)
    async with TestServer(app) as server, http.open():
        endpoint = str(server.make_url("/asr").with_scheme("ws"))
        provider = STT(api_key="key", model_endpoint=endpoint, language="pt", temperature=0.3)
        stream = provider.stream()

        setup = await setups.get()
        await stream.aclose()

    assert setup["json_config"] == {"language": "pt", "temp": 0.3}
