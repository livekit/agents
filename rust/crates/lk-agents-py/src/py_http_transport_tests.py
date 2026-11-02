"""Real local HTTP/WebSocket timeout journeys through the Cargo-linked package."""

import asyncio
import socket
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APITimeoutError
from livekit.agents.utils import http_context

pytestmark = pytest.mark.e2e

_UNLIMITED = APIConnectOptions(
    max_retry=0, pool_timeout=None, connect_timeout=None, write_timeout=None, read_timeout=None
)


@pytest.fixture
async def peer() -> AsyncIterator[tuple[str, asyncio.Event, asyncio.Event]]:
    """A local server that answers each request once ``respond`` is set; it never reads a body."""
    respond, accepted = asyncio.Event(), asyncio.Event()
    handlers: list[asyncio.Task[None]] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            accepted.set()
            await respond.wait()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        handlers.append(asyncio.create_task(handle(reader, writer)))

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", respond, accepted
    finally:
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        server.close()
        await server.wait_closed()


async def test_pool_wait_expires_under_the_pool_limit(
    peer: tuple[str, asyncio.Event, asyncio.Event],
) -> None:
    url, respond, accepted = peer
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:
        holder = asyncio.create_task(_get(session, url, _UNLIMITED))
        await accepted.wait()
        with pytest.raises(APITimeoutError) as error:
            await _get(session, url, replace(_UNLIMITED, pool_timeout=0.05))
        assert error.value.phase == "pool"
        respond.set()
        await holder


async def test_connection_setup_expires_under_the_connect_limit() -> None:
    class StalledResolver(aiohttp.abc.AbstractResolver):
        async def resolve(self, *args: Any, **kwargs: Any) -> Any:
            await asyncio.Event().wait()

        async def close(self) -> None:
            pass

    connector = aiohttp.TCPConnector(resolver=StalledResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        with pytest.raises(APITimeoutError) as error:
            await _get(session, "http://peer.invalid", replace(_UNLIMITED, connect_timeout=0.05))
        assert error.value.phase == "connect"


async def test_stalled_body_upload_expires_under_the_write_limit(
    peer: tuple[str, asyncio.Event, asyncio.Event],
) -> None:
    url, _, _ = peer
    async with aiohttp.ClientSession() as session:
        with pytest.raises(APITimeoutError) as error:
            async with http_context.request(
                session,
                "POST",
                url,
                conn_options=replace(_UNLIMITED, write_timeout=0.2),
                data=bytes(64 * 1024 * 1024),
            ) as response:
                await response.read()
        assert error.value.phase == "write"


async def test_slow_response_is_not_a_write_timeout(
    peer: tuple[str, asyncio.Event, asyncio.Event],
) -> None:
    url, respond, _ = peer
    asyncio.get_running_loop().call_later(0.3, respond.set)
    async with aiohttp.ClientSession() as session:
        async with http_context.request(
            session, "POST", url, conn_options=replace(_UNLIMITED, write_timeout=0.1), data=b"small"
        ) as response:
            assert await response.read() == b"ok"


async def test_stalled_response_expires_under_the_read_limit(
    peer: tuple[str, asyncio.Event, asyncio.Event],
) -> None:
    url, _, _ = peer
    async with aiohttp.ClientSession() as session:
        with pytest.raises(APITimeoutError) as error:
            await _get(session, url, replace(_UNLIMITED, read_timeout=0.05))
        assert error.value.phase == "read"
        assert isinstance(error.value.__cause__, aiohttp.SocketTimeoutError)


async def test_writer_cleanup_preserves_the_expired_phase() -> None:
    async with aiohttp.ClientSession() as session:
        with pytest.raises(APITimeoutError) as error:
            async with http_context._phase_limits(
                session, replace(_UNLIMITED, write_timeout=0), after_connect=None
            ) as phases:
                phases.arm("write")
                try:
                    await asyncio.Event().wait()
                finally:
                    phases.arm(None)
        assert error.value.phase == "write"


async def test_an_unclassified_timeout_keeps_its_identity() -> None:
    failure = TimeoutError("application timeout")
    async with aiohttp.ClientSession() as session:
        with pytest.raises(TimeoutError) as error:
            async with http_context._phase_limits(session, _UNLIMITED, after_connect=None):
                raise failure
        assert error.value is failure


async def test_unanswered_websocket_handshake_expires_under_the_read_limit() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()  # accepts the connection, never answers the upgrade
    port = listener.getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(APITimeoutError) as error:
                await http_context.ws_connect(
                    session,
                    f"ws://127.0.0.1:{port}",
                    conn_options=replace(_UNLIMITED, read_timeout=0.05),
                )
            assert error.value.phase == "read"
    finally:
        listener.close()


async def _get(session: aiohttp.ClientSession, url: str, options: APIConnectOptions) -> bytes:
    async with http_context.request(session, "GET", url, conn_options=options) as response:
        return await response.read()


async def test_a_websocket_peer_that_stops_reading_expires_under_the_write_limit() -> None:
    # a streaming plugin pushes audio to a peer that stopped reading (its buffers full): without
    # the write limit the send waits forever and the stream wedges
    async def handle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await asyncio.Event().wait()  # never receives
        return ws

    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        async with aiohttp.ClientSession() as session:
            ws = await http_context.ws_connect(
                session,
                f"ws://127.0.0.1:{port}/",
                conn_options=replace(_UNLIMITED, write_timeout=0.2),
            )
            try:
                with pytest.raises(APITimeoutError) as error:
                    for _ in range(1024):
                        await ws.send_bytes(bytes(1024 * 1024))
                assert error.value.phase == "write"
            finally:
                await ws.close()  # its close frame is bounded by the same limit
    finally:
        await runner.cleanup()
