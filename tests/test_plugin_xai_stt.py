"""Tests for the xAI STT plugin: half-open socket detection.

Runs a real `SpeechStream` (its real `_run` loop, driven by `_main_task`) against
fake sockets handed out by a fake HttpSession, so the connect kwargs and the
reconnect behaviour are assertable without a network. Same shape as the Deepgram
tests added in #7206 and the Telnyx tests in #7359.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, http

pytestmark = pytest.mark.plugin("xai")

# Retry immediately so a reconnect shows up within the test timeout.
_FAST_RETRY = APIConnectOptions(
    max_retry=3, retry_interval=0.01, connect_timeout=1.0, read_timeout=1.0
)


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

    async def receive(self) -> http.WebSocketMessage:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _HeartbeatTimeoutSocket:
    """A socket whose ping went unanswered: `receive()` raises `APIConnectionError`.

    After a few failures it reports CLOSE, so a recv loop that steps over the error
    only recovers because of that.
    """

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_bytes(self, data: bytes) -> None:
        pass

    async def send_str(self, data: str) -> None:
        pass

    async def receive(self) -> http.WebSocketMessage:
        self.receives += 1
        # yield, so a recv loop that steps over the error instead of ending fails the
        # receive-count assertion rather than starving the event loop
        await asyncio.sleep(0)
        if self.receives > 3:
            return http.WebSocketMessage(http.WebSocketMessageType.CLOSE, "")
        raise APIConnectionError("websocket failed")

    async def close(self) -> None:
        self.closed = True


class _FakeSession:
    """Stands in for the HttpSession: records connect kwargs, hands out sockets."""

    def __init__(self, make_socket) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.sockets: list[Any] = []
        self._make_socket = make_socket
        self.closed = False

    async def ws_connect(self, url: str, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        ws = self._make_socket()
        self.sockets.append(ws)
        return ws


def _stream(session: _FakeSession):
    from livekit.plugins.xai import STT

    instance = STT(api_key="test-key", http_session=session)  # type: ignore[arg-type]
    return instance.stream(conn_options=_FAST_RETRY)


async def test_socket_is_opened_with_a_heartbeat():
    """`heartbeat` defaults to None, so without it the read side of a
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
    """The receive failure has to end the recv loop and surface as a retryable error,
    rather than being logged and stepped over until the socket reports CLOSE."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()
