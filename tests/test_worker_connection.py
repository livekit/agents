"""Connection-failure propagation tests.

When the worker's connection task exhausts ``max_retry`` it raises, and that
failure must surface out of ``AgentServer.run()`` instead of leaving the worker
hanging forever (https://github.com/livekit/agents/issues/6083).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.job import JobContext
from livekit.agents.worker import AgentServer

pytestmark = pytest.mark.unit


def _make_server() -> AgentServer:
    server = AgentServer(
        ws_url="ws://127.0.0.1:1",  # unreachable: connection refused
        api_key="devkey",
        api_secret="devsecret",
        max_retry=0,
        num_idle_processes=0,
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
