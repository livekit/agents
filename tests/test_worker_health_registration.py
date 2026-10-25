"""Health endpoint must report not-ready until RegisterWorkerResponse (issue #8092)."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import AgentServer, JobContext

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_health_returns_503_before_register_response() -> None:
    """GET / must be 503 while LiveKit accepts the WS but never sends RegisterWorkerResponse."""
    fake_port = 18880
    health_port = 18881
    register_requests = 0

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        nonlocal register_requests
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:  # read RegisterWorkerRequest; never reply
            register_requests += 1
        return ws

    app = web.Application()
    app.router.add_get("/agent", agent_ws)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", fake_port)
    await site.start()

    server = AgentServer(
        ws_url=f"ws://127.0.0.1:{fake_port}",
        api_key="devkey",
        api_secret="secret" * 6,
        host="127.0.0.1",
        port=health_port,
        num_idle_processes=0,
        max_retry=50,
    )

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        pass

    registered = asyncio.Event()
    server.on("worker_registered", lambda *_: registered.set())
    run_task = asyncio.create_task(server.run(devmode=False))

    try:
        for _ in range(50):
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.get(f"http://127.0.0.1:{health_port}/") as resp:
                        body = await resp.text()
                        assert resp.status == 503, (resp.status, body)
                        assert "not connected" in body.lower()
                        break
            except (aiohttp.ClientError, OSError):
                await asyncio.sleep(0.1)
        else:
            raise AssertionError("health endpoint never became reachable")

        assert not registered.is_set()
        assert register_requests >= 1
    finally:
        await server.aclose()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        await runner.cleanup()


@pytest.mark.asyncio
async def test_health_ok_when_unregistered() -> None:
    """unregistered=True never connects; / stays a liveness probe (200)."""
    health_port = 18882
    server = AgentServer(host="127.0.0.1", port=health_port, num_idle_processes=0)

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        pass

    pool_started = asyncio.Event()
    process_pool = MagicMock()
    process_pool.start = AsyncMock(side_effect=lambda: pool_started.set())
    process_pool.aclose = AsyncMock()
    process_pool.on = MagicMock()

    with patch("livekit.agents.ipc.proc_pool.ProcPool", return_value=process_pool):
        run_task = asyncio.create_task(server.run(unregistered=True, devmode=True))
        try:
            await asyncio.wait_for(pool_started.wait(), timeout=10)
            for _ in range(50):
                try:
                    async with aiohttp.ClientSession() as http:
                        async with http.get(f"http://127.0.0.1:{health_port}/") as resp:
                            assert resp.status == 200
                            assert await resp.text() == "OK"
                            return
                except (aiohttp.ClientError, OSError):
                    await asyncio.sleep(0.1)
            raise AssertionError("health endpoint never became reachable")
        finally:
            await server.aclose()
            await run_task


async def _run_reconnect_scenario(fake_port: int, health_port: int) -> list[int]:
    """Register once, drop the WS, then never register again. Returns health statuses
    polled for ~2s after the drop."""
    from livekit.protocol import agent

    phase = {"n": 0}

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        phase["n"] += 1
        if phase["n"] == 1:
            # Accept registration once, then drop the socket so the worker retries.
            async for _raw in ws:
                reply = agent.ServerMessage()
                reply.register.worker_id = "W_test_reconnect"
                await ws.send_bytes(reply.SerializeToString())
                await ws.close()
                break
        else:
            # Subsequent attempts: accept WS but never register (stay in connecting).
            async for _ in ws:
                pass
        return ws

    app = web.Application()
    app.router.add_get("/agent", agent_ws)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", fake_port)
    await site.start()

    server = AgentServer(
        ws_url=f"ws://127.0.0.1:{fake_port}",
        api_key="devkey",
        api_secret="secret" * 6,
        host="127.0.0.1",
        port=health_port,
        num_idle_processes=0,
        max_retry=50,
    )

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        pass

    registered = asyncio.Event()
    server.on("worker_registered", lambda *_: registered.set())
    run_task = asyncio.create_task(server.run(devmode=False))

    statuses: list[int] = []
    try:
        await asyncio.wait_for(registered.wait(), timeout=10)
        # Allow the fake server close to propagate into the worker's retry loop.
        await asyncio.sleep(0.5)
        for _ in range(20):
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.get(f"http://127.0.0.1:{health_port}/") as resp:
                        statuses.append(resp.status)
            except (aiohttp.ClientError, OSError):
                pass
            await asyncio.sleep(0.1)
    finally:
        await server.aclose()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        await runner.cleanup()
    return statuses


@pytest.mark.asyncio
async def test_health_ok_during_short_reconnect() -> None:
    """A registered worker that briefly loses its WS must stay healthy (no pod restart)."""
    statuses = await _run_reconnect_scenario(18883, 18884)
    assert statuses and all(s == 200 for s in statuses), statuses


@pytest.mark.asyncio
async def test_health_returns_503_after_reconnect_grace_period() -> None:
    """Once disconnected for longer than the grace period, GET / must report 503."""
    with patch("livekit.agents.worker.HEALTH_RECONNECT_GRACE_PERIOD", 0.5):
        statuses = await _run_reconnect_scenario(18885, 18886)
    assert 503 in statuses, statuses
    assert statuses[-1] == 503, statuses
