"""Neuphonic TTS plugin: SSE audio parsing."""

from __future__ import annotations

import base64
import json

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.neuphonic import TTS


async def test_null_sse_audio_keeps_synthesis_successful() -> None:
    """An SSE event with null audio is skipped; the synthesis still succeeds."""
    pcm = b"\x01\x00" * 480
    requests = 0

    async def speak(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(
            text='data: {"data": {"audio": null}}\n\n'
            + "data: "
            + json.dumps({"data": {"audio": base64.b64encode(pcm).decode()}})
            + "\n\n",
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/sse/speak/en", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="test", base_url=str(server.make_url("/")).rstrip("/"), http_session=session
        )
        try:
            async with provider.synthesize(
                "hello", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                samples = sum([event.frame.samples_per_channel async for event in stream])
            assert samples >= len(pcm) // 2
            assert requests == 1
        finally:
            await provider.aclose()
