"""Server drain and shutdown through the native runtime and host callback boundary."""

import asyncio

import pytest

from livekit.agents import _core
from livekit.agents._generated.worker_options import ProcessOptions, ServerOptions
from livekit.agents.http import AgentRoutes

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_a_server_without_job_processes_drains_and_closes() -> None:
    class Host:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def _core_start(self, options: object) -> None:
            return None

        async def _core_started(self) -> None:
            self.started.set()

        async def _core_heartbeat(self) -> None:
            pass

    host = Host()
    config = _core.ServerConfig(ServerOptions(), ProcessOptions(), "0.0.0")
    config.simulation = True
    runtime = _core.ServerRuntime(config, host)
    running = asyncio.ensure_future(
        runtime.run(
            devmode=True,
            unregistered=True,
            has_rtc=False,
            has_text_doors=True,
            routes=[],
            has_load_fnc=False,
            endpoints=AgentRoutes(),
            text_doors=[],
        )
    )
    try:
        await asyncio.wait_for(host.started.wait(), timeout=5)
        assert runtime.active_jobs == []
        await asyncio.wait_for(runtime.drain(), timeout=5)
        await asyncio.wait_for(runtime.aclose(), timeout=5)
        await asyncio.wait_for(running, timeout=1)
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
