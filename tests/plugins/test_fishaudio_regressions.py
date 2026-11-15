"""Fish Audio socket reuse regression."""

import msgpack  # type: ignore[import-untyped]  # msgpack ships no type information
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.fishaudio import TTS


async def test_a_short_reply_flushes_on_finish_and_the_next_reuses_its_socket() -> None:
    """A one-chunk reply plays whole when Fish finishes it, and the next reply reuses the socket.

    https://github.com/livekit/agents/pull/6368 (its cold-start prebuffer was dropped by
    https://github.com/livekit/agents/pull/6423: playout starts on the first chunk)
    """
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
        try:
            for _ in range(2):
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    stream.push_text("Hello.")
                    stream.end_input()
                    frames = [ev async for ev in stream]
                assert b"".join(bytes(ev.frame.data) for ev in frames) == pcm
            assert connections == 1
        finally:
            await provider.aclose()
