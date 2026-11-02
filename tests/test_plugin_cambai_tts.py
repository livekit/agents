"""Camb.ai TTS: `speaking_rate` is sent as `voice_settings.speaking_rate` only when set."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions

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


async def _synthesize(tts) -> None:
    stream = tts.synthesize("hello world", conn_options=APIConnectOptions(max_retry=0, timeout=5))
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


async def test_speaking_rate_omitted_by_default():
    from livekit.plugins.cambai import TTS

    async with _Server() as server:
        tts = TTS(api_key="test-key", base_url=server.base_url, http_session=server.session)
        await _synthesize(tts)

    assert "voice_settings" not in server.payloads[0]


async def test_speaking_rate_from_constructor():
    from livekit.plugins.cambai import TTS

    async with _Server() as server:
        tts = TTS(
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
            speaking_rate=1.5,
        )
        await _synthesize(tts)

    assert server.payloads[0]["voice_settings"] == {"speaking_rate": 1.5}


async def test_update_options_speaking_rate_applies_to_next_request():
    from livekit.plugins.cambai import TTS

    async with _Server() as server:
        tts = TTS(
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
            speaking_rate=1.5,
        )
        await _synthesize(tts)
        tts.update_options(speaking_rate=0.8)
        await _synthesize(tts)

    assert server.payloads[0]["voice_settings"] == {"speaking_rate": 1.5}
    assert server.payloads[1]["voice_settings"] == {"speaking_rate": 0.8}


async def test_update_options_speaking_rate_none_resets_to_default():
    from livekit.plugins.cambai import TTS

    async with _Server() as server:
        tts = TTS(
            api_key="test-key",
            base_url=server.base_url,
            http_session=server.session,
            speaking_rate=1.5,
        )
        tts.update_options(speaking_rate=None)
        await _synthesize(tts)

    assert "voice_settings" not in server.payloads[0]


def test_update_options_without_speaking_rate_keeps_it():
    from livekit.plugins.cambai import TTS

    tts = TTS(api_key="test-key", speaking_rate=1.2)
    tts.update_options(voice_id=1234)
    assert tts._opts.speaking_rate == 1.2
