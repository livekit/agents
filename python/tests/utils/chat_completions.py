"""A local chat-completions endpoint that replays one SSE stream and records each request."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from aiohttp import web


@dataclass
class Received:
    headers: dict[str, str]
    """By lowercase name."""
    body: bytes
    """Decompressed: aiohttp undoes a gzip ``Content-Encoding``."""


@contextlib.asynccontextmanager
async def serve(stream: bytes) -> AsyncIterator[tuple[str, list[Received]]]:
    """Yields the endpoint's ``base_url`` and the requests it received."""
    received: list[Received] = []

    async def complete(request: web.Request) -> web.Response:
        headers = {name.lower(): value for name, value in request.headers.items()}
        received.append(Received(headers, await request.read()))
        return web.Response(body=stream, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/v1", received
    finally:
        await runner.cleanup()
