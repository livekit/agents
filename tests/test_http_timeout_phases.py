import asyncio
from collections.abc import AsyncIterator

import aiohttp
import pytest

from livekit.agents import APIConnectOptions, APIError
from livekit.agents.utils.http_context import client_timeout

pytestmark = pytest.mark.e2e


@pytest.fixture
async def stalled_response_url() -> AsyncIterator[str]:
    handlers: list[asyncio.Task[None]] = []

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        handlers.append(asyncio.create_task(respond(reader, writer)))

    async with await asyncio.start_server(accept, "127.0.0.1", 0) as server:
        try:
            yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        finally:
            for handler in handlers:
                handler.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["read", "total"])
async def test_stalled_response_expires_only_under_the_enabled_limit(
    stalled_response_url: str, phase: str
) -> None:
    options = APIConnectOptions(
        max_retry=0,
        pool_timeout=None,
        connect_timeout=None,
        write_timeout=None,
        read_timeout=0.05 if phase == "read" else None,
        total_timeout=0.05 if phase == "total" else None,
    )
    async with aiohttp.ClientSession() as session:
        with pytest.raises(TimeoutError):
            async with session.get(
                stalled_response_url, timeout=client_timeout(options)
            ) as response:
                await response.read()


@pytest.mark.parametrize("phase", ["pool", "connect", "write"])
def test_unobservable_phase_is_refused_before_a_request(phase: str) -> None:
    options = APIConnectOptions(
        pool_timeout=0.1 if phase == "pool" else None,
        connect_timeout=0.1 if phase == "connect" else None,
        write_timeout=0.1 if phase == "write" else None,
    )
    with pytest.raises(APIError, match=f"conn_options.{phase}_timeout") as error:
        client_timeout(options)
    assert not error.value.retryable


def test_total_limit_cannot_reset_on_another_http_attempt() -> None:
    options = APIConnectOptions(
        max_retry=1,
        pool_timeout=None,
        connect_timeout=None,
        write_timeout=None,
        total_timeout=1,
    )
    with pytest.raises(APIError, match="across retries") as error:
        client_timeout(options)
    assert not error.value.retryable
