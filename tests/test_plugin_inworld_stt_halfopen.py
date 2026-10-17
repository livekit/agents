"""Tests for Inworld STT half-open socket detection (#7900)."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import pytest

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.inworld import stt as inworld_stt

pytestmark = pytest.mark.unit


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream"
        await asyncio.sleep(0.01)


class _ParkedSocket:
    """A socket that accepts writes and never delivers anything: the half-open case
    from the client's point of view. `receive()` only returns once the socket is
    closed, and the test never lets that happen."""

    def __init__(self) -> None:
        self.closed = False
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        pass

    async def __aiter__(self):
        # AsyncIterator for the `async for msg in ws` pattern
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")
        yield  # unreachable, but needed for typing

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _HeartbeatTimeoutSocket:
    """A socket in the state aiohttp leaves it in when a ping goes unanswered.

    The heartbeat closes the connection itself, so this arrives as WSMsgType.ERROR
    rather than as a close frame, and the reason lives only on `exception()`. After a
    few ERROR frames it reports CLOSED, which is what aiohttp does on the next pass;
    a recv loop that steps over the ERROR only recovers because of that.
    """

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_str(self, data: str) -> None:
        pass

    def exception(self) -> BaseException:
        return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

    async def __aiter__(self):
        # AsyncIterator for the `async for msg in ws` pattern
        while True:
            msg = await self.receive()
            yield msg
            if msg.type == aiohttp.WSMsgType.CLOSED:
                break

    async def receive(self) -> aiohttp.WSMessage:
        self.receives += 1
        # yield, so a recv loop that steps over the error instead of ending fails the
        # receive-count assertion rather than starving the event loop
        await asyncio.sleep(0)
        if self.receives > 3:
            return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)
        return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), None)

    async def close(self) -> None:
        self.closed = True


class _FakeSession:
    """Stands in for the aiohttp session: records connect kwargs, hands
    out sockets, and can be closed."""

    def __init__(self, make_socket) -> None:
        self.closed = False
        self.kwargs: list[dict[str, Any]] = []
        self.sockets: list[Any] = []
        self._make_socket = make_socket

    async def ws_connect(self, url: str, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        ws = self._make_socket()
        self.sockets.append(ws)
        return ws

    async def close(self) -> None:
        self.closed = True


def _stream(session: _FakeSession) -> inworld_stt.SpeechStream:
    """Create a SpeechStream with a fake session."""
    stt = inworld_stt.STT(api_key="sk_test", base_url="http://localhost:8000/", http_session=session)  # type: ignore[arg-type]
    return stt.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)


@pytest.mark.asyncio
async def test_socket_is_opened_with_a_heartbeat() -> None:
    """aiohttp defaults `heartbeat` to None, so without it the read side of a
    half-open socket parks forever and the reconnect in `_run`, which only fires
    when `recv_task` ends, never gets a turn."""
    session = _FakeSession(_ParkedSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.kwargs) == 1)
        assert session.kwargs[0].get("heartbeat") == 30.0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_heartbeat_timeout_reconnects_without_spinning() -> None:
    """The ERROR frame has to end the recv loop and request a reconnect. Logging it
    as an unexpected type and continuing only reconnected because aiohttp happened
    to report CLOSED next, and it threw away the one value that says why the socket
    went."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()
