"""Fake aiohttp websockets for exercising STT plugins' connection handling.

`ParkedSocket` is the half-open case (#7900): writes succeed and nothing is ever
delivered. `HeartbeatTimeoutSocket` is the state aiohttp leaves a socket in when a
heartbeat ping goes unanswered. `FakeWSSession` stands in for the aiohttp session.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp


async def wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the stream"
        await asyncio.sleep(0.01)


class _FakeSocket:
    def __init__(self) -> None:
        self.closed = False
        self._response = type("obj", (object,), {"headers": {}})()

    async def send_json(self, data: Any) -> None:
        pass

    async def send_str(self, data: str) -> None:
        pass

    async def receive(self) -> aiohttp.WSMessage:
        raise NotImplementedError

    async def __aiter__(self) -> AsyncIterator[aiohttp.WSMessage]:
        # mirrors aiohttp: iteration stops once the socket reports a closing message
        while True:
            msg = await self.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                break
            yield msg

    async def close(self) -> None:
        self.closed = True


class ParkedSocket(_FakeSocket):
    """Accepts writes and never delivers anything: the half-open case from the
    client's point of view. `receive()` only returns once the socket is closed, and
    the tests never let that happen."""

    def __init__(self) -> None:
        super().__init__()
        self._closed = asyncio.Event()

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        raise AssertionError("the test should never let recv_task resume")

    async def close(self) -> None:
        await super().close()
        self._closed.set()


class HeartbeatTimeoutSocket(_FakeSocket):
    """A socket whose heartbeat ping went unanswered.

    aiohttp closes the connection itself, so this arrives as WSMsgType.ERROR rather
    than as a close frame, and the reason lives only on `exception()`. After a few
    ERROR frames it reports CLOSED, which is what aiohttp does on the next pass; a
    recv loop that steps over the ERROR only recovers because of that.
    """

    def __init__(self) -> None:
        super().__init__()
        self.receives = 0

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


class FakeWSSession:
    """Records `ws_connect` kwargs and hands out sockets from `make_socket`."""

    def __init__(self, make_socket: Callable[[], _FakeSocket]) -> None:
        self.closed = False
        self.kwargs: list[dict[str, Any]] = []
        self.sockets: list[_FakeSocket] = []
        self._make_socket = make_socket

    async def ws_connect(self, url: str, **kwargs: Any) -> _FakeSocket:
        self.kwargs.append(kwargs)
        ws = self._make_socket()
        self.sockets.append(ws)
        return ws

    async def close(self) -> None:
        self.closed = True
