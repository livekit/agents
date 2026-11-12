"""Fireworks AI plugin: streaming STT connection options."""

from __future__ import annotations

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.fireworksai import STT


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/8368 reproduces: stream(language=) is ignored",
)
async def test_stream_language_applies_to_that_stream_only() -> None:
    """A per-stream language reaches only that socket; https://github.com/livekit/agents/issues/8368."""
    languages: list[str | None] = []

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        languages.append(request.query.get("language"))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/{path:.*}", transcribe)
    async with TestServer(app) as server, http.open() as session:
        model = STT(
            api_key="test",
            language="en",
            base_url=str(server.make_url("/v1").with_scheme("ws")),
            http_session=session,
        )
        options = APIConnectOptions(max_retry=0)
        try:
            for stream in (model.stream(language="fr", conn_options=options), model.stream()):
                async with stream:
                    stream.end_input()
                    async for _ in stream:
                        pass
        finally:
            await model.aclose()

    assert languages == ["fr", "en"]
