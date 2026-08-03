from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from livekit.agents.ipc.job_proc_lazy_main import _JobProc, _ShutdownInfo
from livekit.agents.ipc.proto import InitializeRequest
from livekit.agents.job import JobContext, JobExecutorType
from livekit.agents.worker import AgentServer, ServerOptions

pytestmark = [pytest.mark.unit]


def test_server_options_entrypoint_shutdown_timeout() -> None:
    opts = ServerOptions(
        entrypoint_fnc=lambda ctx: None,  # type: ignore
        entrypoint_shutdown_timeout=2.5,
    )
    server = AgentServer.from_server_options(opts)
    assert server._entrypoint_shutdown_timeout == 2.5


@pytest.mark.asyncio
async def test_job_proc_entrypoint_shutdown_timeout_cancels() -> None:
    entrypoint_cancelled = False

    async def slow_entrypoint(ctx: JobContext) -> None:
        nonlocal entrypoint_cancelled
        try:
            await asyncio.sleep(100.0)
        except asyncio.CancelledError:
            entrypoint_cancelled = True
            raise

    job_proc = _JobProc(
        initialize_process_fnc=lambda proc: None,
        job_entrypoint_fnc=slow_entrypoint,
        session_end_fnc=None,
        session_end_timeout=5.0,
        entrypoint_shutdown_timeout=0.1,
        executor_type=JobExecutorType.THREAD,
    )

    fake_client = AsyncMock()
    job_proc.initialize(InitializeRequest(), fake_client)  # type: ignore

    class FakeRunningJob:
        fake_job = True

        class Job:
            id = "test_job"
            agent_name = "test_agent"

            class Room:
                name = "test_room"

            room = Room()

        job = Job()

    class FakeStartReq:
        running_job = FakeRunningJob()

    job_proc._exit_proc_flag = asyncio.Event()
    job_proc._shutdown_fut = asyncio.Future[_ShutdownInfo]()

    # Start job task
    job_proc._start_job(FakeStartReq())  # type: ignore
    assert job_proc._job_task is not None

    # Simulate shutdown signal
    job_proc._shutdown_fut.set_result(_ShutdownInfo(user_initiated=False, reason="test shutdown"))

    # Wait for the job task to complete (it should finish in ~0.1s due to entrypoint_shutdown_timeout)
    start_time = asyncio.get_running_loop().time()
    await job_proc._job_task
    elapsed = asyncio.get_running_loop().time() - start_time

    assert elapsed < 2.0, f"Expected entrypoint cancellation in ~0.1s, took {elapsed:.2f}s"
    assert entrypoint_cancelled is True
