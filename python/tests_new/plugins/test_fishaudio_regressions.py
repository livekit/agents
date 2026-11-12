"""Fish Audio cold-start delivery regression."""

import asyncio

import msgpack  # type: ignore[import-untyped]  # msgpack ships no type information
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.fishaudio import TTS


@pytest.mark.parametrize(
    "chunks",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/pull/6368 reproduces: first chunk plays before cold-start prebuffer",
            ),
        ),
    ],
)
async def test_cold_start_prebuffer_and_short_final_flush(chunks: int) -> None:
    """Cold starts buffer two chunks while short replies flush and reuse sockets. https://github.com/livekit/agents/pull/6368"""
    first_sent, release = asyncio.Event(), asyncio.Event()
    connections = 0
    pcm = b"\x03\x00" * 2400

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal connections
        connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = msgpack.unpackb(msg.data, raw=False)
            if packet["event"] == "stop":
                await ws.send_bytes(
                    msgpack.packb({"event": "audio", "audio": pcm}, use_bin_type=True)
                )
                first_sent.set()
                if chunks == 2:
                    await release.wait()
                    await ws.send_bytes(
                        msgpack.packb({"event": "audio", "audio": pcm}, use_bin_type=True)
                    )
                await ws.send_bytes(
                    msgpack.packb({"event": "finish", "reason": "stop"}, use_bin_type=True)
                )
        return ws

    app = web.Application()
    app.router.add_get("/v1/tts/live", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            output_format="pcm",
            sample_rate=24000,
            base_url=str(server.make_url("")),
            http_session=session,
        )
        premature = False
        try:
            for attempt in range(2):
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    stream.push_text("Hello.")
                    stream.end_input()
                    first = asyncio.create_task(anext(stream))
                    try:
                        await asyncio.wait_for(first_sent.wait(), 1)
                        if chunks == 2 and attempt == 0:
                            done, _ = await asyncio.wait({first}, timeout=0.1)
                            premature = bool(done)
                        release.set()
                        head = await asyncio.wait_for(first, 1)
                        frames = [head, *[ev async for ev in stream]]
                    finally:
                        release.set()
                        if not first.done():
                            first.cancel()
                        await asyncio.gather(first, return_exceptions=True)
                assert b"".join(bytes(ev.frame.data) for ev in frames) == pcm * chunks
            assert connections == 1
            assert not premature, "first chunk played while the second chunk was still blocked"
        finally:
            release.set()
            await provider.aclose()
