"""Tests for the Gladia STT plugin: half-open socket detection.

Runs a real `SpeechStream` (its real `_run` loop, driven by `_main_task`) against
fake sockets handed out by a fake aiohttp session, so the connect kwargs and the
reconnect behaviour are assertable without a network. Same shape as the Deepgram
tests added in #7206 and the Telnyx tests added in #7359.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.plugin("gladia")

# Retry immediately so a reconnect shows up within the test timeout.
_FAST_RETRY = APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0)


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream"
        await asyncio.sleep(0.01)


class _ParkedSocket:
    """A socket that accepts writes and never delivers anything: the half-open case
    from the client's point of view. The message iterator only ends once the socket
    is closed, and the test never lets that happen."""

    def __init__(self) -> None:
        self.closed = False
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    def __aiter__(self) -> _ParkedSocket:
        return self

    async def __anext__(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _HeartbeatTimeoutSocket:
    """A socket in the state aiohttp leaves it in when a ping goes unanswered.

    The heartbeat closes the connection itself, so this arrives as WSMsgType.ERROR
    rather than as a close frame, and the reason lives only on `exception()`. After a
    few ERROR frames the iterator ends, which is what aiohttp does on the next pass;
    a recv loop that steps over the ERROR only recovers because of that.
    """

    def __init__(self) -> None:
        self.closed = False
        self.receives = 0

    async def send_str(self, data: str) -> None:
        pass

    async def send_bytes(self, data: bytes) -> None:
        pass

    def exception(self) -> BaseException:
        return aiohttp.ServerTimeoutError("No PONG received after 15.0 seconds")

    def __aiter__(self) -> _HeartbeatTimeoutSocket:
        return self

    async def __anext__(self) -> aiohttp.WSMessage:
        self.receives += 1
        # yield, so a recv loop that steps over the error instead of ending fails the
        # receive-count assertion rather than starving the event loop
        await asyncio.sleep(0)
        if self.receives > 3:
            raise StopAsyncIteration
        return aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, self.exception(), None)

    async def close(self) -> None:
        self.closed = True


class _FakeResponse:
    """The `POST /v2/live` response that hands out the session URL."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.status = 201
        self._payload = payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> dict[str, Any]:
        return self._payload


class _FakeWSContext:
    """`ws_connect` is used as an async context manager by the Gladia plugin."""

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def __aenter__(self) -> Any:
        return self._ws

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


class _FakeSession:
    """Stands in for the aiohttp session: records connect kwargs, hands out sockets."""

    def __init__(self, make_socket) -> None:
        self.closed = False
        self.kwargs: list[dict[str, Any]] = []
        self.sockets: list[Any] = []
        self._make_socket = make_socket

    def post(self, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse({"id": "test-session", "url": "wss://example.invalid/live"})

    def ws_connect(self, url: str, **kwargs: Any) -> _FakeWSContext:
        self.kwargs.append(kwargs)
        ws = self._make_socket()
        self.sockets.append(ws)
        return _FakeWSContext(ws)

    async def close(self) -> None:
        self.closed = True


def _stream(session: _FakeSession):
    from livekit.plugins.gladia import STT

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
    Logging it as an unexpected message type and continuing only reconnected because
    aiohttp happened to end the iterator next, and it threw away the one value that
    says why the socket went."""
    session = _FakeSession(_HeartbeatTimeoutSocket)
    stream = _stream(session)
    try:
        await _wait_until(lambda: len(session.sockets) > 1)
        assert session.sockets[0].receives == 1
    finally:
        await stream.aclose()
