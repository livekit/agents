import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, APITimeoutError
from livekit.agents.utils import http_context

pytestmark = pytest.mark.unit

_UNLIMITED = APIConnectOptions(
    max_retry=0, pool_timeout=None, connect_timeout=None, write_timeout=None, read_timeout=None
)


def _only(phase: str, limit: float = 0.05) -> APIConnectOptions:
    return APIConnectOptions(
        max_retry=0,
        **{
            f"{p}_timeout": limit if p == phase else None
            for p in ("pool", "connect", "write", "read")
        },
    )


@pytest.fixture
async def peer() -> AsyncIterator[tuple[str, asyncio.Event]]:
    """A local server that answers each request once ``respond`` is set; it never reads a body."""
    respond = asyncio.Event()
    handlers: list[asyncio.Task[None]] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
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
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", respond
    finally:
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        server.close()


async def test_request_completes_under_the_shipped_defaults(
    peer: tuple[str, asyncio.Event],
) -> None:
    url, respond = peer
    respond.set()
    async with aiohttp.ClientSession() as session:
        async with http_context.request(
            session, "POST", url, conn_options=DEFAULT_API_CONNECT_OPTIONS, json={"text": "hello"}
        ) as response:
            assert await response.read() == b"ok"


async def test_pool_wait_expires_under_the_pool_limit(peer: tuple[str, asyncio.Event]) -> None:
    url, respond = peer
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:
        holder = asyncio.create_task(_get(session, url, _UNLIMITED))
        await asyncio.sleep(0.05)
        with pytest.raises(APITimeoutError) as error:
            await _get(session, url, _only("pool"))
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
            await _get(session, "http://peer.invalid", _only("connect"))
        assert error.value.phase == "connect"


async def test_stalled_body_upload_expires_under_the_write_limit(
    peer: tuple[str, asyncio.Event],
) -> None:
    url, _ = peer
    async with aiohttp.ClientSession() as session:
        with pytest.raises(APITimeoutError) as error:
            async with http_context.request(
                session, "POST", url, conn_options=_only("write", 0.2), data=bytes(64 * 1024 * 1024)
            ) as response:
                await response.read()
        assert error.value.phase == "write"


async def test_slow_response_is_not_a_write_timeout(peer: tuple[str, asyncio.Event]) -> None:
    url, respond = peer
    asyncio.get_running_loop().call_later(0.3, respond.set)
    async with aiohttp.ClientSession() as session:
        async with http_context.request(
            session, "POST", url, conn_options=_only("write", 0.1), data=b"small"
        ) as response:
            assert await response.read() == b"ok"


async def test_stalled_response_expires_under_the_read_limit(
    peer: tuple[str, asyncio.Event],
) -> None:
    url, _ = peer
    async with aiohttp.ClientSession() as session:
        with pytest.raises(TimeoutError):
            await _get(session, url, _only("read"))


async def test_unanswered_websocket_handshake_expires_under_the_read_limit() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()  # accepts the connection, never answers the upgrade
    port = listener.getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(APITimeoutError) as error:
                await http_context.ws_connect(
                    session, f"ws://127.0.0.1:{port}", conn_options=_only("read")
                )
            assert error.value.phase == "read"
    finally:
        listener.close()


async def _get(session: aiohttp.ClientSession, url: str, options: APIConnectOptions) -> bytes:
    async with http_context.request(session, "GET", url, conn_options=options) as response:
        return await response.read()
