"""Offline transports for exercising provider connection boundaries with captured audio."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import stt


async def wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.sent: list[bytes | str | dict] = []
        self.closed = False
        self.close_code = None
        self._response = SimpleNamespace(headers={})

    def __await__(self):
        async def connect():
            return self

        return connect().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.receive()
        if message.type == aiohttp.WSMsgType.CLOSED:
            raise StopAsyncIteration
        return message

    async def receive(self):
        return await self.incoming.get()

    def respond(self, data: dict) -> None:
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(data), ""))

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True
        self.incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))


class ReconnectClock:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, ready: dict | None = None) -> None:
        self.now = 1000.0
        self.sockets: list[FakeWebSocket] = []
        self.ready = ready
        self.closed = False
        monkeypatch.setattr(time, "time", lambda: self.now)

    def ws_connect(self, *args: Any, **kwargs: Any) -> FakeWebSocket:
        ws = FakeWebSocket()
        if self.ready is not None:
            ws.respond(self.ready)
        self.sockets.append(ws)
        return ws

    async def close(self) -> None:
        self.closed = True

    async def push(self, stream: stt.RecognizeStream, *, captured_at: float) -> None:
        sample_rate = stream._needed_sr or 16000
        self.now = captured_at + 1.0
        stream.push_frame(rtc.AudioFrame.create(sample_rate, 1, sample_rate))
        await wait_until(lambda: stream._input_ch.empty() or stream._task.done())
        if stream._task.done():
            stream._task.result()
            pytest.fail("stream ended before consuming audio")

    async def check(self, stream: stt.RecognizeStream, reconnect: Callable[[], None]) -> None:
        stream.start_time_offset = 7.0
        await self.push(stream, captured_at=1000.0)
        assert stream.start_time_offset == pytest.approx(7.0)
        self.now = 1010.0
        reconnect()
        await wait_until(lambda: len(self.sockets) == 2 or stream._task.done())
        await self.push(stream, captured_at=1010.0)
        assert stream.start_time_offset == pytest.approx(17.0)
