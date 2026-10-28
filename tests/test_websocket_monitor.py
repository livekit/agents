from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError
from livekit.agents.utils.aio import WebSocketMonitor

pytestmark = pytest.mark.unit


@pytest.mark.virtual_time
@pytest.mark.parametrize("liveness_timeout", [None, 3.0])
async def test_liveness_timeout_controls_probe_and_failure(liveness_timeout: float | None) -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    pings: list[float] = []

    async def ping() -> None:
        pings.append(loop.time() - started)

    ws = AsyncMock()
    ws.receive.side_effect = asyncio.Event().wait
    ws.ping.side_effect = ping
    options = {} if liveness_timeout is None else {"liveness_timeout": liveness_timeout}
    with pytest.raises(APIConnectionError):
        await WebSocketMonitor(cast(aiohttp.ClientWebSocketResponse, ws), **options).receive()
    timeout = 6.0 if liveness_timeout is None else liveness_timeout
    assert pings == pytest.approx([timeout / 3])
    assert loop.time() - started == pytest.approx(timeout)


@contextlib.asynccontextmanager
async def gateway(
    handler: Callable[[web.Request], Any],
) -> AsyncIterator[aiohttp.ClientWebSocketResponse]:
    app = web.Application()
    app.router.add_get("/", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        async with session.ws_connect(server.make_url("/"), autoping=False) as ws:
            yield ws


async def test_silence_stays_healthy_when_peer_answers_pings() -> None:
    pings = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal pings
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.PING:
                pings += 1
                await ws.pong(msg.data)
                if pings == 5:
                    await ws.send_str("still connected")
        return ws

    async with gateway(handler) as ws:
        msg = await asyncio.wait_for(WebSocketMonitor(ws, liveness_timeout=0.2).receive(), 2)
        assert msg.data == "still connected"
        assert not ws.closed
    assert pings == 5


@pytest.mark.parametrize("message_type", ["text", "binary", "ping", "pong"])
async def test_inbound_messages_reset_deadline_without_client_pings(message_type: str) -> None:
    client_pings = 0
    server_pongs = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal client_pings, server_pongs
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)

        async def send_messages() -> None:
            for _ in range(15):
                await asyncio.sleep(0.02)
                if message_type == "text":
                    await ws.send_str("activity")
                elif message_type == "binary":
                    await ws.send_bytes(b"activity")
                elif message_type == "ping":
                    await ws.ping(b"activity")
                else:
                    await ws.pong(b"activity")
            await ws.send_str("done")

        sender = asyncio.create_task(send_messages())
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.PING:
                    client_pings += 1
                elif msg.type == aiohttp.WSMsgType.PONG:
                    server_pongs += 1
        finally:
            await sender
        return ws

    async with gateway(handler) as ws:
        monitor = WebSocketMonitor(ws, liveness_timeout=0.2)
        while (msg := await asyncio.wait_for(monitor.receive(), 2)).data != "done":
            assert msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY)
    assert client_pings == 0
    if message_type == "ping":
        assert server_pongs == 15


async def test_outbound_audio_does_not_hide_unanswered_ping() -> None:
    pings = 0
    audio_frames = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal pings, audio_frames
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.PING:
                pings += 1
            elif msg.type == aiohttp.WSMsgType.BINARY:
                audio_frames += 1
        return ws

    async with gateway(handler) as ws:

        async def send_audio() -> None:
            while True:
                await ws.send_bytes(b"audio")
                await asyncio.sleep(0.01)

        sender = asyncio.create_task(send_audio())
        try:
            with pytest.raises(APIConnectionError) as exc_info:
                await asyncio.wait_for(WebSocketMonitor(ws, liveness_timeout=0.2).receive(), 2)
            assert exc_info.value.retryable
            assert isinstance(exc_info.value.__cause__, aiohttp.ServerTimeoutError)
            assert ws.closed
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
    assert pings == 1
    assert audio_frames > 1


async def test_data_after_ping_resets_pending_failure() -> None:
    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        await asyncio.sleep(0.12)
        await ws.send_str("first")
        await asyncio.sleep(0.12)
        await ws.send_str("second")
        async for _ in ws:
            pass
        return ws

    async with gateway(handler) as ws:
        monitor = WebSocketMonitor(ws, liveness_timeout=0.2)
        assert (await monitor.receive()).data == "first"
        assert (await monitor.receive()).data == "second"
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(monitor.receive(), 2)


async def test_cancelled_receive_does_not_reset_silence_deadline() -> None:
    ws = AsyncMock()
    ws.receive.side_effect = asyncio.Event().wait
    monitor = WebSocketMonitor(cast(aiohttp.ClientWebSocketResponse, ws), liveness_timeout=0.2)
    for _ in range(3):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(monitor.receive(), 0.05)
    with pytest.raises(APIConnectionError):
        await asyncio.wait_for(monitor.receive(), 0.1)
    assert ws.ping.await_count == 1


async def test_blocked_ping_write_cannot_delay_failure() -> None:
    ws = AsyncMock()
    ws.receive.side_effect = asyncio.Event().wait
    ws.ping.side_effect = asyncio.Event().wait
    with pytest.raises(APIConnectionError):
        await asyncio.wait_for(
            WebSocketMonitor(
                cast(aiohttp.ClientWebSocketResponse, ws), liveness_timeout=0.2
            ).receive(),
            1,
        )
    ws.close.assert_awaited_once()


async def test_unresponsive_close_cannot_delay_failure() -> None:
    release = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoclose=False, autoping=False)
        await ws.prepare(request)
        await release.wait()
        return ws

    async with gateway(handler) as ws:
        try:
            with pytest.raises(APIConnectionError) as exc_info:
                await asyncio.wait_for(WebSocketMonitor(ws, liveness_timeout=0.2).receive(), 1)
            assert isinstance(exc_info.value.__cause__, aiohttp.ServerTimeoutError)
            assert ws.closed
        finally:
            release.set()
