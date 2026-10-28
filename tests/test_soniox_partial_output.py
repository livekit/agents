"""A later provider failure never restarts a synthesis that already delivered audio."""

from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.plugins.soniox import TTS

pytestmark = pytest.mark.unit


async def test_a_failure_after_audio_is_not_replayed() -> None:
    requests: list[str] = []
    first_received = asyncio.Event()

    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.type != aiohttp.WSMsgType.TEXT:
                continue
            message = incoming.json()
            if not message.get("text"):
                continue
            requests.append(message["text"])
            stream_id = message["stream_id"]
            if len(requests) == 1:
                await ws.send_json(
                    {
                        "stream_id": stream_id,
                        "audio": base64.b64encode(b"\x00\x00" * 2400).decode(),
                    }
                )
                await ws.send_json({"stream_id": stream_id, "audio_end": True, "terminated": True})
                first_received.set()
            else:
                await ws.send_json(
                    {"stream_id": stream_id, "error_code": 429, "error_message": "later failure"}
                )
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
                api_key="test", websocket_url=f"ws://127.0.0.1:{port}/tts", http_session=session
            )
            try:
                options = APIConnectOptions(max_retry=1, retry_interval=0)
                async with tts.stream(conn_options=options) as stream:
                    stream.push_text("First sentence is long enough to start synthesis. Next ")
                    # Wait for delivered audio, not just the peer's send acknowledgement.
                    async with asyncio.timeout(2):
                        audio = await anext(stream)
                        assert audio.frame.samples_per_channel > 0
                        await first_received.wait()
                        stream.push_text("sentence must report the provider failure. ")
                        stream.end_input()
                        with pytest.raises(APIStatusError, match="later failure"):
                            async for _ in stream:
                                pass
                assert len(requests) == 2
            finally:
                await tts.aclose()
    finally:
        await runner.cleanup()
