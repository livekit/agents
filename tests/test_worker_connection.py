"""Connection-failure propagation tests.

When the worker's connection task exhausts ``max_retry`` it raises, and that
failure must surface out of ``AgentServer.run()`` instead of leaving the worker
hanging forever (https://github.com/livekit/agents/issues/6083).
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from livekit.agents.inference_runner import _InferenceRunner
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


class _FakeInferenceExecutor:
    """Stand-in for InferenceProcExecutor.

    The real executor spawns a child process plus a non-daemon
    ``proc_join_thread`` that keeps the interpreter alive until it is closed.
    """

    thread_name = "fake_inference_proc_join_thread"

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        self.started = True
        # deliberately non-daemon: like the real proc_join_thread it must be
        # joined by aclose() or the process never exits
        self._thread = threading.Thread(target=self._stop.wait, name=self.thread_name)
        self._thread.start()

    async def initialize(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True
        self.started = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def is_alive(self) -> bool:
        return self.started and self._thread is not None and self._thread.is_alive()


async def test_aclose_closes_resources_started_before_run_raises() -> None:
    """A startup failure after the inference executor starts must not leak it.

    Regression: ``_close_future`` was only created after the HTTP/Prometheus
    servers and the proc pool were up. A bind failure in that window (the
    default port 8081 already being used) raised out of ``run()`` while
    ``_closed`` was already ``False``; ``aclose()`` then saw
    ``_close_future is None`` and returned early without closing the inference
    executor. Its child process and non-daemon join thread kept the CLI alive.
    """
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]

    server = AgentServer(
        ws_url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="secretsecretsecretsecretsecretsecret",
        port=port,
        num_idle_processes=0,
    )

    @server.rtc_session()
    async def _entry(ctx: JobContext) -> None:
        pass

    fake_executor = _FakeInferenceExecutor()

    try:
        # run() builds the executor itself; make the registry non-empty so it
        # takes that path, and substitute the fake for the real process.
        with (
            patch.dict(_InferenceRunner.registered_runners, {"test_dummy": object()}, clear=False),
            patch(
                "livekit.agents.worker.ipc.inference_proc_executor.InferenceProcExecutor",
                return_value=fake_executor,
            ),
        ):
            with pytest.raises(OSError):
                await asyncio.wait_for(server.run(devmode=True), timeout=10)

        assert fake_executor.started, (
            "inference executor should have started before the port bind failed"
        )

        started_at = time.monotonic()
        await asyncio.wait_for(server.aclose(), timeout=10)
        elapsed = time.monotonic() - started_at

        assert fake_executor.closed, "aclose() must close the inference executor"
        assert not fake_executor.is_alive()
        live_threads = [
            t.name for t in threading.enumerate() if t.name == fake_executor.thread_name
        ]
        assert not live_threads, f"non-daemon executor thread still alive: {live_threads}"
        http_server = getattr(server, "_http_server", None)
        assert http_server is not None
        assert http_server._runner is None, "HTTP server should be cleaned up"
        assert elapsed < 5, f"aclose() took too long: {elapsed:.2f}s"
    finally:
        # keep the test process able to exit when the regression is present
        if not fake_executor.closed:
            await fake_executor.aclose()
        http_server = getattr(server, "_http_server", None)
        if http_server is not None:
            await http_server.aclose()
        blocker.close()
