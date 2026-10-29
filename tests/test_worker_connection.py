"""A failed registration settles the public server run and keeps close idempotent."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import AgentHandler, AgentServer, JobContext

pytestmark = pytest.mark.unit


async def test_run_raises_when_connection_exhausts_retries() -> None:
    handler = AgentHandler()

    @handler.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        pass

    server = AgentServer(
        ws_url="ws://127.0.0.1:1",
        api_key="devkey",
        api_secret="devsecret",
        num_idle_processes=0,
    )
    server._update_options(max_retry=0, port=0)
    server.add(handler)
    try:
        with pytest.raises(RuntimeError, match="connection failed after 0 retries"):
            await asyncio.wait_for(server._run(devmode=True), timeout=10)
    finally:
        await server.aclose()
        await server.aclose()
