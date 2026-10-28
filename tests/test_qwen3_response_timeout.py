"""Qwen3 times expected synthesis progress without timing the application's text gaps."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APITimeoutError
from livekit.plugins.baseten import TTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["stalled", "input_gap", "audio_progress"])
async def test_read_limit_tracks_pending_synthesis(mode: str) -> None:
    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.type != aiohttp.WSMsgType.TEXT:
                continue
            if incoming.json()["type"] != "input.done" or mode == "stalled":
                continue
            await ws.send_json({"type": "audio.start", "sample_rate": 24000})
            for _ in range(6):
                await ws.send_bytes(b"\x00\x00" * 2400)
                if mode == "audio_progress":
                    await asyncio.sleep(0.02)
            await ws.send_json({"type": "audio.done"})
            await ws.send_json({"type": "session.done"})
        return ws

    app = web.Application()
    app.router.add_get("/tts", synthesize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(
                model="qwen3-tts",
                api_key="test",
                voice="test",
                model_endpoint=f"ws://127.0.0.1:{port}/tts",
                http_session=session,
            )
            try:
                options = APIConnectOptions(max_retry=0, read_timeout=0.05)
                async with tts.stream(conn_options=options) as stream:
                    stream.push_text("Hello there.")
                    if mode == "input_gap":
                        # No input.done has been sent, so no synthesis reply is outstanding.
                        await asyncio.sleep(0.12)
                    stream.end_input()
                    async with asyncio.timeout(0.8):
                        if mode == "stalled":
                            with pytest.raises(APITimeoutError) as error:
                                async for _ in stream:
                                    pass
                            assert error.value.phase == "read"
                        else:
                            assert (
                                sum([audio.frame.samples_per_channel async for audio in stream]) > 0
                            )
            finally:
                await tts.aclose()
    finally:
        await runner.cleanup()
