"""Server drain and shutdown through the native runtime and host callback boundary."""

import asyncio

import pytest

from livekit.agents import _core

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_close_during_drain_does_not_wait_for_running_jobs() -> None:
    class Host:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.joining = asyncio.Event()
            self.closed = asyncio.Event()

        async def open(self, kind: str, args: object) -> object:
            if kind == "started":
                self.started.set()
            elif kind == "join_jobs":
                self.joining.set()
                unfinished: asyncio.Future[None] = asyncio.Future()
                await unfinished
            elif kind == "close":
                self.closed.set()
            return None

    host = Host()
    config = _core.ServerConfig({})
    config.simulation = True
    runtime = _core.ServerRuntime(config, host)
    running = asyncio.ensure_future(
        runtime.run(
            devmode=True,
            unregistered=True,
            has_rtc=False,
            has_text_doors=True,
            publisher=False,
            routes=[],
            has_load_fnc=False,
            has_on_request=False,
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
