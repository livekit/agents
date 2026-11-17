"""Camb.ai TTS: `speaking_rate` is sent as `voice_settings.speaking_rate` only when set."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions
from livekit.plugins.cambai import TTS

pytestmark = pytest.mark.unit


class _Server:
    """Local ``/tts-stream`` stand-in that records each request body."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def _handler(self, request: web.Request) -> web.Response:
        self.payloads.append(await request.json())
        return web.Response(body=b"\x00\x00" * 2205)

    async def __aenter__(self) -> _Server:
        app = web.Application()
        app.router.add_post("/tts-stream", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()


@pytest.mark.parametrize(
    ("speaking_rate", "update", "expected"),
    [
        (None, {}, None),
        (1.5, {}, 1.5),
        (1.5, {"speaking_rate": 0.8}, 0.8),
        (1.5, {"speaking_rate": None}, None),
        (1.2, {"voice_id": 1234}, 1.2),
    ],
)
async def test_speaking_rate_sent_only_when_set(speaking_rate, update, expected):
    async with _Server() as server:
        tts = TTS(
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
            speaking_rate=speaking_rate,
        )
        tts.update_options(**update)
        async for _ in tts.synthesize(
            "hello world", conn_options=APIConnectOptions(max_retry=0, timeout=5)
        ):
            pass

    voice_settings = server.payloads[0].get("voice_settings")
    assert voice_settings == (None if expected is None else {"speaking_rate": expected})
