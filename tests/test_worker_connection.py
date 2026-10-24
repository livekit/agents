"""Connection-failure propagation tests.

When the worker's connection task exhausts ``max_retry`` it raises, and that
failure must surface out of ``AgentServer.run()`` instead of leaving the worker
hanging forever (https://github.com/livekit/agents/issues/6083).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents.job import JobContext
from livekit.agents.worker import AgentServer
from livekit.protocol import agent

pytestmark = pytest.mark.unit


def _make_server() -> AgentServer:
    server = AgentServer(
        ws_url="ws://127.0.0.1:1",  # unreachable: connection refused
        api_key="devkey",
        api_secret="devsecret",
        max_retry=0,
        num_idle_processes=0,
        http_proxy=None,
    )

    @server.rtc_session()
    async def _entry(ctx: JobContext) -> None:
        pass

    server._simulation = True  # skip binding the health HTTP server
    return server


async def test_run_raises_when_connection_exhausts_retries() -> None:
    server = _make_server()

    fake_pool = MagicMock()
    fake_pool.start = AsyncMock()
    fake_pool.aclose = AsyncMock()
    fake_pool.processes = []

    with patch("livekit.agents.ipc.proc_pool.ProcPool", return_value=fake_pool):
        try:
            with pytest.raises(RuntimeError, match="failed to connect"):
                await asyncio.wait_for(server.run(devmode=True), timeout=10)
        finally:
            await server.aclose()
            await server.aclose()  # repeated aclose() stays a no-op


async def test_health_requires_worker_registration() -> None:
    register_received = asyncio.Event()
    allow_registration = asyncio.Event()
    worker_registered = asyncio.Event()

    async def fake_agent(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        message = agent.WorkerMessage()
        message.ParseFromString(await ws.receive_bytes())
        assert message.HasField("register")
        register_received.set()

        await allow_registration.wait()
        response = agent.ServerMessage(
            register=agent.RegisterWorkerResponse(worker_id="test-worker")
        )
        await ws.send_bytes(response.SerializeToString())
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/agent", fake_agent)
    runner = web.AppRunner(app)
    await runner.setup()
    server = AgentServer(
        ws_url="ws://127.0.0.1",
        api_key="devkey",
        api_secret="devsecret",
        host="127.0.0.1",
        port=0,
        num_idle_processes=0,
        http_proxy=None,
    )

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        pass

    server.on("worker_registered", lambda *_: worker_registered.set())
    run_task: asyncio.Task[None] | None = None
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        server.update_options(ws_url=f"ws://127.0.0.1:{runner.addresses[0][1]}")
        run_task = asyncio.create_task(server.run(devmode=True))
        await asyncio.wait_for(register_received.wait(), timeout=10)
        assert not worker_registered.is_set()
        assert server._http_server is not None
        health_url = f"http://127.0.0.1:{server._http_server.port}/"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(health_url) as response:
                assert response.status == 503
                assert await response.text() == "not connected to livekit"
            assert not worker_registered.is_set()

            allow_registration.set()
            await asyncio.wait_for(worker_registered.wait(), timeout=10)
            async with session.get(health_url) as response:
                assert response.status == 200
                assert await response.text() == "OK"
    finally:
        try:
            await server.aclose()
        finally:
            if run_task is not None:
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            allow_registration.set()
            await runner.cleanup()
