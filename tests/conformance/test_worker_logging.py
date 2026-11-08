"""Supervisor diagnostics reach Python's logging without blocking job execution."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
from typing import Any

import pytest

from livekit.agents import JobContext, _core
from livekit.agents._generated.worker_server import AcceptArguments, JobInfo, RunningJobInfo
from livekit.agents.ipc import JobRunnerOptions
from livekit.agents.ipc.child import JobArgs, _JobProc
from livekit.agents.job import JobExecutorType
from livekit.protocol import agent, models

pytestmark = pytest.mark.unit


def test_thread_job_exit_logs_preserve_fields_with_a_stalled_handler() -> None:
    # A fresh interpreter must exercise the first supervisor, before any server/process launch
    # has initialized process-wide services.
    result = subprocess.run(
        [sys.executable, __file__, "--worker"], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


_FIELDS = ("reason", "job_id", "room", "room_id", "pid")


async def _run_job() -> None:
    entered = threading.Event()
    release = threading.Event()
    records: list[tuple[int, str, dict[str, Any]]] = []

    class Stalling(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if message == "job thread initialized":
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release the log handler")
            fields = {key: getattr(record, key) for key in _FIELDS if hasattr(record, key)}
            records.append((record.levelno, message, fields))

    agents_logger = logging.getLogger("livekit.agents")
    agents_logger.addHandler(Stalling(logging.DEBUG))
    agents_logger.setLevel(logging.DEBUG)

    class Parent:
        def __init__(self) -> None:
            self.endpoint: asyncio.Future[int] = asyncio.get_running_loop().create_future()
            self.launch: _core.JobLaunch | None = None

        async def _core_launch(self, endpoint: int, launch: _core.JobLaunch) -> None:
            self.launch = launch
            self.endpoint.set_result(endpoint)

        async def _core_completed(self, completion: Any) -> None:
            pass

    async def entrypoint(ctx: JobContext) -> None:
        ctx.shutdown("done ✓")

    runner = _JobProc(
        JobArgs(
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=None,
            simulation_end_fnc=None,
            session_end_timeout=1,
        ),
        JobExecutorType.THREAD,
    )
    parent = Parent()
    supervisor = _core.JobSupervisor(
        parent,
        JobRunnerOptions(
            thread=True,
            initialize_timeout=3,
            close_timeout=3,
            ping_interval=0,
            ping_timeout=0,
            high_ping_threshold=1,
        ),
    )
    await supervisor.start()
    child_fd = await parent.endpoint
    initializing = asyncio.ensure_future(supervisor.initialize())
    client = await asyncio.to_thread(_core.IpcClient, child_fd, runner.initialize)
    runner._client = client
    serving = asyncio.ensure_future(client.serve(runner))
    try:
        await initializing
        assert await asyncio.to_thread(entered.wait, 3), records
        # The log drain is blocked in the handler. Job acceptance and shutdown must still
        # complete on the real IPC connection and Python loop.
        job = agent.Job(id="AJ_logging", room=models.Room(name="logging"))
        info = RunningJobInfo(
            accept_arguments=AcceptArguments(name="", identity="agent", metadata="", attributes=[]),
            job=JobInfo(
                id=job.id,
                dispatch_id="",
                type=0,
                room="logging",
                room_sid=None,
                agent_name="",
                metadata="",
                deployment="",
                enable_recording=False,
                enable_redaction=False,
                proto=job.SerializeToString(),
            ),
            url="",
            token="",
            worker_id="worker",
            fake_job=True,
            received_at=None,
            accepted_at=None,
            assigned_at=None,
        )
        await asyncio.wait_for(supervisor.launch_job(info), timeout=3)
        await asyncio.wait_for(asyncio.shield(serving), timeout=3)
        assert runner._job_task is not None
        await asyncio.wait_for(asyncio.shield(runner._job_task), timeout=3)
        assert not release.is_set()
        release.set()
        async with asyncio.timeout(3):
            while not any(message == "job thread exiting" for _, message, _ in records):
                await asyncio.sleep(0.01)
        # the supervisor's diagnostics name the job the runner runs
        expected = {"reason": "done ✓", "job_id": "AJ_logging", "room": "logging"}
        assert (logging.INFO, "job thread exiting", expected) in records, records
    finally:
        release.set()
        await client.aclose()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        await runner.aclose()
        assert parent.launch is not None
        parent.launch.thread_exited()


if __name__ == "__main__":
    asyncio.run(_run_job())
