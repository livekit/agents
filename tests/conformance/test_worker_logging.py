"""Supervisor diagnostics cross the job IPC boundary without blocking job execution."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
from typing import Any

import pytest

from livekit.agents import JobContext, JobExecutorType, _core
from livekit.agents.ipc.child import _JobProc
from livekit.protocol import agent, models

pytestmark = pytest.mark.unit


def test_thread_job_exit_logs_preserve_fields_with_a_stalled_handler() -> None:
    # A fresh interpreter must exercise the first supervisor, before any server/process launch
    # has initialized process-wide services.
    result = subprocess.run(
        [sys.executable, __file__, "--worker"], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _run_job() -> None:
    entered = threading.Event()
    release = threading.Event()
    records: list[tuple[int, str, dict[str, Any]]] = []

    class Parent:
        def _core_log(
            self, level: int, message: str, extra: dict[str, Any], exc: Any = None
        ) -> None:
            if message == "job runner initialized":
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release the diagnostic writer")
            records.append((level, message, extra))

    async def entrypoint(ctx: JobContext) -> None:
        ctx.shutdown("done ✓")

    runner = _JobProc(entrypoint, None, session_end_timeout=1, executor_type=JobExecutorType.THREAD)
    parent_fd, child_fd = _core.ipc_pair()
    supervisor = _core.ProcSupervisor(
        parent_fd,
        Parent(),
        pid=0,
        initialize_timeout=3,
        close_timeout=3,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=0,
        ping_timeout=0,
        high_ping_threshold=1,
        killable=False,
    )
    initializing = asyncio.ensure_future(
        supervisor.initialize(asyncio_debug=False, http_proxy=None)
    )
    client = await asyncio.to_thread(_core.IpcClient, child_fd, runner.initialize)
    runner._client = client
    serving = asyncio.ensure_future(client.serve(runner))
    try:
        await initializing
        assert await asyncio.to_thread(entered.wait, 3), records
        # The writer is blocked in the host handler. Job acceptance and shutdown must still
        # complete on the real IPC connection and Python loop.
        await asyncio.wait_for(
            supervisor.start_job(
                _core.StartJobFields(
                    job=agent.Job(
                        id="AJ_logging", room=models.Room(name="logging")
                    ).SerializeToString(),
                    name="",
                    identity="agent",
                    metadata="",
                    url="",
                    token="",
                    worker_id="worker",
                    fake_job=True,
                    received_at=0,
                    accepted_at=0,
                    assigned_at=0,
                    launched_at=0,
                )
            ),
            timeout=3,
        )
        await asyncio.wait_for(asyncio.shield(serving), timeout=3)
        assert runner._job_task is not None
        await asyncio.wait_for(asyncio.shield(runner._job_task), timeout=3)
        assert not release.is_set()
        release.set()
        async with asyncio.timeout(3):
            while not any(message == "job exiting" for _, message, _ in records):
                await asyncio.sleep(0.01)
        assert (logging.DEBUG, "job exiting", {"reason": "done ✓"}) in records, records
    finally:
        release.set()
        await client.aclose()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        await runner.aclose()


if __name__ == "__main__":
    asyncio.run(_run_job())
