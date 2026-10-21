"""Tests for the xAI STT plugin: half-open socket detection.

Runs a real `SpeechStream` (its real `_run` loop, driven by `_main_task`) against
fake sockets handed out by a fake aiohttp session, so the connect kwargs and the
reconnect behaviour are assertable without a network. Same shape as the Deepgram
tests added in #7206 and the Telnyx tests in #7359.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.plugin("xai")

# Retry immediately so a reconnect shows up within the test timeout.
_FAST_RETRY = APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0)


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

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def send_str(self, data: str) -> None:
        pass

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

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def send_str(self, data: str) -> None:
        pass

    def exception(self) -> BaseException:
        return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

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
    """Stands in for the aiohttp session: records connect kwargs, hands out sockets."""

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


def _stream(session: _FakeSession):
    from livekit.plugins.xai import STT

    instance = STT(api_key="test-key", http_session=session)  # type: ignore[arg-type]
    return instance.stream(conn_options=_FAST_RETRY)


async def test_socket_is_opened_with_a_heartbeat():
    """aiohttp defaults `heartbeat` to None, so without it the read side of a
    half-open socket parks forever and the retry in `_main_task`, which only runs
    when something raises, never gets a turn."""
    session = _FakeSession(_ParkedSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.kwargs) == 1)
        assert session.kwargs[0].get("heartbeat") == 30.0
    finally:
        await stream.aclose()


async def test_heartbeat_timeout_reconnects_without_spinning():
    """The ERROR frame has to end the recv loop and surface as a retryable error.
    Logging it and continuing only reconnected because aiohttp happened to report
    CLOSED next, and it threw away the one value that says why the socket went."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()
