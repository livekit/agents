"""Server drain and shutdown through the native runtime and host callback boundary."""

import asyncio

import pytest

from livekit.agents import _core
from livekit.agents._generated.worker_options import ServerOptions
from livekit.agents.http import AgentRoutes, AsgiApplication

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_close_during_drain_does_not_wait_for_running_jobs() -> None:
    class Host:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.joining = asyncio.Event()
            self.closed = asyncio.Event()

        async def _core_start(self, options: object) -> None:
            pass

        async def _core_started(self) -> None:
            self.started.set()

        async def _core_worker_load(self, load: float) -> None:
            pass

        async def _core_join_jobs(self) -> None:
            self.joining.set()
            unfinished: asyncio.Future[None] = asyncio.Future()
            await unfinished

        async def _core_close(self) -> None:
            self.closed.set()

    host = Host()
    config = _core.ServerConfig(ServerOptions())
    config.simulation = True
    runtime = _core.ServerRuntime(config, host)
    http = AgentRoutes()
    running = asyncio.ensure_future(
        runtime.run(
            devmode=True,
            unregistered=True,
            has_rtc=False,
            has_text_doors=True,
            routes=[],
            has_load_fnc=False,
            endpoints=http._server,
            app=AsgiApplication(http.app),
        )
    )
    draining: asyncio.Future[object] | None = None
    try:
        await asyncio.wait_for(host.started.wait(), timeout=5)
        draining = asyncio.ensure_future(runtime.drain())
        await asyncio.wait_for(host.joining.wait(), timeout=5)
        await asyncio.wait_for(runtime.aclose(), timeout=1)
        assert host.closed.is_set()
        await asyncio.wait_for(running, timeout=1)
        with pytest.raises(RuntimeError, match="server closed during drain"):
            await asyncio.wait_for(draining, timeout=1)
    finally:
        if draining is not None:
            if not draining.done():
                draining.cancel()
            await asyncio.gather(draining, return_exceptions=True)
        try:
            await asyncio.wait_for(runtime.aclose(), timeout=2)
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
