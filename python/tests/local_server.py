"""An ``AgentServer`` served in process, as ``lk agent console`` serves it: unregistered, its
routes and text doors on the worker's own loopback listener, where every caller holds every
permission. Authentication through a node or a bearer is pinned by the door's Rust tests."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from livekit.agents import AgentServer


@contextlib.asynccontextmanager
async def serving(server: AgentServer) -> AsyncIterator[str]:
    """The listener's base URL, ``http://127.0.0.1:<port>/agents/<agent>/<deployment>``, while
    ``server`` runs."""
    server._update_options(listeners={"health": {"port": 0}})
    run = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        url = None
        for _ in range(200):
            with contextlib.suppress(RuntimeError):
                url = server._config.delegation_url("").rstrip("/")
                break
            if run.done():
                run.result()
            await asyncio.sleep(0.05)
        if url is None:
            raise TimeoutError("the public listener did not start")
        yield url
    finally:
        await server.aclose()
        await run
