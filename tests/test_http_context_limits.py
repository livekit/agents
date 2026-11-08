"""Each ``APIConnectOptions`` limit fires on the wait it names, against loopback peers."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APITimeoutError
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils import http_context

pytestmark = pytest.mark.unit

FAST = 0.2
SLOW = 5.0


def _options(**limits: float | None) -> APIConnectOptions:
    every = {"pool_timeout": SLOW, "connect_timeout": SLOW, "write_timeout": SLOW}
    return APIConnectOptions(max_retry=0, **(every | {"read_timeout": SLOW} | limits))


@pytest.fixture
async def server() -> AsyncIterator[str]:
    """``/slow?delay=`` answers late, ``/hang`` never answers, ``/endless`` never finishes its
    body, ``/sink`` never reads the request body and ``/ws`` never reads frames."""
    release = asyncio.Event()

    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(float(request.query["delay"]))
        return web.Response(text="ok")

    async def hang(request: web.Request) -> web.Response:
        await release.wait()
        return web.Response()

    async def endless(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await release.wait()
        return response

    async def ws_idle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.transport.pause_reading()  # type: ignore[union-attr]
        await release.wait()
        return ws

    app = web.Application(client_max_size=1 << 30)
    app.router.add_route("*", "/slow", slow)
    app.router.add_get("/hang", hang)
    app.router.add_get("/endless", endless)
    app.router.add_post("/sink", hang)
    app.router.add_get("/ws", ws_idle)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        release.set()
        await runner.cleanup()


@pytest.fixture
async def silent_peer() -> AsyncIterator[str]:
    """Accepts connections and never answers."""
    held: list[asyncio.StreamWriter] = []

    async def accept(_: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    srv = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        for w in held:
            w.close()
        srv.close()


async def _timeout_of(call: Callable[[], object]) -> str | None:
    """The limit ``call`` fails with; a limit that never fires fails the test after 3s."""
    with pytest.raises(APITimeoutError) as raised:
        async with asyncio.timeout(3):
            await call()  # type: ignore[misc]
    return raised.value.phase


async def _get(session: aiohttp.ClientSession, url: str, options: APIConnectOptions) -> None:
    async with http_context.request(session, "GET", url, conn_options=options) as resp:
        await resp.read()


async def test_a_slow_response_fires_the_read_limit(server: str) -> None:
    async with aiohttp.ClientSession() as session:
        phase = await _timeout_of(
            lambda: _get(session, f"{server}/hang", _options(read_timeout=FAST))
        )
    assert phase == "read"


async def test_a_response_within_the_read_limit_ignores_the_other_limits(server: str) -> None:
    """Waiting for the response is not a write, pool or connect wait."""
    options = _options(read_timeout=SLOW, write_timeout=0.05, pool_timeout=0.05, connect_timeout=1)
    async with aiohttp.ClientSession() as session:
        async with http_context.request(
            session, "POST", f"{server}/slow?delay=0.3", conn_options=options, data=b"x"
        ) as resp:
            assert await resp.text() == "ok"


async def test_a_slow_dns_lookup_fires_the_connect_limit(server: str) -> None:
    class SlowResolver(aiohttp.ThreadedResolver):
        async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):  # type: ignore[no-untyped-def,override]
            await asyncio.sleep(SLOW)
            return await super().resolve(host, port, family)

    connector = aiohttp.TCPConnector(resolver=SlowResolver(), use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        url = server.replace("127.0.0.1", "localhost") + "/hang"
        phase = await _timeout_of(lambda: _get(session, url, _options(connect_timeout=FAST)))
    assert phase == "connect"


async def test_waiting_for_a_pooled_connection_fires_the_pool_limit(server: str) -> None:
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with http_context.request(
            session, "GET", f"{server}/endless", conn_options=_options()
        ):  # holds the only connection
            phase = await _timeout_of(
                lambda: _get(session, f"{server}/hang", _options(pool_timeout=FAST))
            )
    assert phase == "pool"


async def test_a_body_the_peer_does_not_read_fires_the_write_limit(server: str) -> None:
    async def post() -> None:
        async with http_context.request(
            session,
            "POST",
            f"{server}/sink",
            conn_options=_options(write_timeout=FAST),
            data=b"x" * (256 << 20),
        ):
            pass

    async with aiohttp.ClientSession() as session:
        phase = await _timeout_of(post)
    assert phase == "write"


async def test_a_silent_websocket_handshake_fires_the_read_limit(silent_peer: str) -> None:
    async with aiohttp.ClientSession() as session:
        phase = await _timeout_of(
            lambda: http_context.ws_connect(
                session, f"{silent_peer}/ws", conn_options=_options(read_timeout=FAST)
            )
        )
    assert phase == "read"


async def test_frames_the_peer_does_not_read_fire_the_write_limit(server: str) -> None:
    async with aiohttp.ClientSession() as session:
        ws = await http_context.ws_connect(
            session, f"{server}/ws", conn_options=_options(write_timeout=FAST)
        )

        async def flood() -> None:
            while True:
                await ws.send_bytes(b"x" * (1 << 20))

        try:
            phase = await _timeout_of(flood)
        finally:
            ws._writer.transport.abort()  # type: ignore[union-attr]
    assert phase == "write"


async def test_the_job_session_applies_the_limits_too(server: str) -> None:
    async with http_context.open() as session:
        phase = await _timeout_of(
            lambda: _get(session, f"{server}/hang", _options(read_timeout=FAST))
        )
    assert phase == "read"
