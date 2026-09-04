from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.job import JobContext
from livekit.agents.worker import AgentServer

pytestmark = pytest.mark.unit


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(),
    reason="forkserver is not available on this platform",
)
async def test_agent_server_runs_freeze_preload_last() -> None:
    server = AgentServer(multiprocessing_context="forkserver", num_idle_processes=0)

    @server.rtc_session()
    async def entrypoint(_: JobContext) -> None:
        pass

    server._simulation = True
    server._mp_ctx = mp_context = MagicMock()

    pool_started = asyncio.Event()
    process_pool = MagicMock()
    process_pool.start = AsyncMock(side_effect=lambda: pool_started.set())
    process_pool.aclose = AsyncMock()

    with patch("livekit.agents.ipc.proc_pool.ProcPool", return_value=process_pool):
        run_task = asyncio.create_task(server.run(unregistered=True))
        try:
            await asyncio.wait_for(pool_started.wait(), timeout=5)
            preloads = mp_context.set_forkserver_preload.call_args.args[0]

            assert preloads[-1] == "livekit.agents.ipc._preload_freeze"
        finally:
            await server.aclose()
            await run_task


@pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods(),
    reason="forkserver is not available on this platform",
)
def test_forkserver_preload_preserves_child_gc() -> None:
    helper = Path(__file__).parent / "utils" / "forkserver_preload.py"
    result = subprocess.run(
        [sys.executable, str(helper)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    outcome = json.loads(result.stdout)

    assert outcome["freeze_count"] > 0
    assert outcome["child_cycle_collected"] is True
