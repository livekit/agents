"""Connection-failure propagation tests.

When the worker's connection task exhausts ``max_retry`` it raises, and that
failure must surface out of ``AgentServer.run()`` instead of leaving the worker
hanging forever (https://github.com/livekit/agents/issues/6083).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from livekit.agents.job import JobAcceptArguments, JobContext, RunningJobInfo
from livekit.agents.worker import AgentServer
from livekit.protocol import agent

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


async def test_reports_running_jobs_as_assigned_after_registering() -> None:
    server = _make_server()

    # the job as the availability request carried it: the identity is the one
    # the worker accepted it with
    job = agent.Job(id="AJ_running", state=agent.JobState(worker_id="AW_test"))
    running = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="test", identity="agent-identity", metadata=""),
        job=job,
        url="ws://localhost",
        token="test-token",
        worker_id="AW_test",
        fake_job=False,
    )

    with (
        patch.object(AgentServer, "active_jobs", new_callable=PropertyMock, return_value=[running]),
        patch.object(server, "_queue_msg", new_callable=AsyncMock) as queue_msg,
    ):
        await server._report_active_jobs()

    req = queue_msg.call_args.args[0].migrate_job
    assert list(req.job_ids) == ["AJ_running"]
    assert len(req.jobs) == 1
    assert req.jobs[0].id == "AJ_running"
    assert req.jobs[0].state.worker_id == "AW_test"
    assert req.jobs[0].state.participant_identity == "agent-identity"
    assert job.state.participant_identity == "", "the running job is not changed"
