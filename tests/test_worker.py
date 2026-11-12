from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import AgentServer, JobContext, JobExecutorType
from livekit.agents.cli import proto
from livekit.agents.cli.cli import _run_worker
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.mock_room import create_mock_room
from livekit.agents.ipc.proc_pool import ProcPool
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.agents.worker import _is_cloud_url, _pick_region_url
from livekit.protocol import agent, models

pytestmark = pytest.mark.unit


async def _noop_entrypoint(ctx: JobContext) -> None:
    pass


@contextlib.asynccontextmanager
async def _fake_livekit(
    agent_ws: Callable[[web.Request], Awaitable[web.StreamResponse]],
    **routes: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[int]:
    """Serve the worker websocket (and extra POST routes) on a free port."""
    app = web.Application()
    app.router.add_get("/agent", agent_ws)
    for path, handler in routes.items():
        app.router.add_post(f"/{path}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    await web.SockSite(runner, sock).start()
    try:
        yield sock.getsockname()[1]
    finally:
        await runner.cleanup()


def _registered_server(port: int, health_port: int) -> AgentServer:
    server = AgentServer(
        ws_url=f"ws://127.0.0.1:{port}",
        api_key="devkey",
        api_secret="secret" * 6,
        host="127.0.0.1",
        port=health_port,
        num_idle_processes=0,
        max_retry=50,
    )
    server.rtc_session()(_noop_entrypoint)
    return server


async def _health(health_port: int) -> tuple[int, str] | None:
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(f"http://127.0.0.1:{health_port}/") as resp:
                return resp.status, await resp.text()
    except (aiohttp.ClientError, OSError):
        return None


# -- config precedence: url/keys are CLI > constructor > env; agent_name is
# LIVEKIT_AGENT_NAME_OVERRIDE > rtc_session(agent_name=) > LIVEKIT_AGENT_NAME


class _CapturingServer(AgentServer):
    async def run(self, **kwargs: Any) -> None:
        self.captured_config = {
            "ws_url": self._ws_url,
            "api_key": self._api_key,
            "api_secret": self._api_secret,
            "agent_name": self._agent_name,
        }
        raise SystemExit(0)


@pytest.mark.parametrize(
    ("env", "ctor", "decorator_name", "cli", "expected"),
    [
        (
            {"LIVEKIT_URL": "ws://env", "LIVEKIT_API_KEY": "env-key", "LIVEKIT_API_SECRET": "e"},
            {"ws_url": "ws://ctor", "api_key": "ctor-key", "api_secret": "c"},
            "",
            {"url": "ws://cli", "api_key": "cli-key", "api_secret": "cli-secret"},
            {"ws_url": "ws://cli", "api_key": "cli-key", "api_secret": "cli-secret"},
        ),
        ({"LIVEKIT_URL": "ws://env"}, {"ws_url": "ws://ctor"}, "", {}, {"ws_url": "ws://ctor"}),
        ({"LIVEKIT_URL": "ws://env"}, {}, "", {}, {"ws_url": "ws://env"}),
        (
            {"LIVEKIT_AGENT_NAME_OVERRIDE": "override", "LIVEKIT_AGENT_NAME": "env"},
            {},
            "decorator",
            {},
            {"agent_name": "override"},
        ),
        ({"LIVEKIT_AGENT_NAME": "env"}, {}, "decorator", {}, {"agent_name": "decorator"}),
        ({"LIVEKIT_AGENT_NAME": "env"}, {}, "", {}, {"agent_name": "env"}),
        ({}, {}, "", {}, {"agent_name": ""}),
    ],
)
def test_config_precedence(
    env: dict[str, str],
    ctor: dict[str, str],
    decorator_name: str,
    cli: dict[str, str],
    expected: dict[str, str],
) -> None:
    with patch.dict(os.environ, env, clear=True):
        server = _CapturingServer(**ctor)  # type: ignore[arg-type]
        server.rtc_session(agent_name=decorator_name)(_noop_entrypoint)
        with contextlib.suppress(SystemExit):
            _run_worker(server, proto.CliArgs(log_level="INFO", **cli))

    assert {k: server.captured_config[k] for k in expected} == expected


# -- connection


async def test_run_raises_when_connection_exhausts_retries() -> None:
    server = AgentServer(
        ws_url="ws://127.0.0.1:1",  # connection refused
        api_key="devkey",
        api_secret="devsecret",
        max_retry=0,
        num_idle_processes=0,
    )
    server.rtc_session()(_noop_entrypoint)
    server._simulation = True  # skip binding the health HTTP server

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
    server = AgentServer(api_key="devkey", api_secret="devsecret", num_idle_processes=0)
    # the identity is the one the worker accepted the job with
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
    assert [j.id for j in req.jobs] == ["AJ_running"]
    assert req.jobs[0].state.worker_id == "AW_test"
    assert req.jobs[0].state.participant_identity == "agent-identity"
    assert job.state.participant_identity == "", "the running job is not changed"


class _FakeInferenceExecutor:
    """Stand-in for InferenceProcExecutor, with the same non-daemon join thread."""

    thread_name = "fake_inference_proc_join_thread"

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        self.started = True
        self._thread = threading.Thread(target=self._stop.wait, name=self.thread_name)
        self._thread.start()

    async def initialize(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


async def test_aclose_closes_resources_started_before_run_raises() -> None:
    """A port bind failure after the inference executor started must not leak the executor."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)

    server = AgentServer(
        ws_url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="secretsecretsecretsecretsecretsecret",
        port=blocker.getsockname()[1],
        num_idle_processes=0,
    )
    server.rtc_session()(_noop_entrypoint)
    fake_executor = _FakeInferenceExecutor()

    try:
        with (
            patch.dict(_InferenceRunner.registered_runners, {"test_dummy": object()}),
            patch(
                "livekit.agents.worker.ipc.inference_proc_executor.InferenceProcExecutor",
                return_value=fake_executor,
            ),
        ):
            with pytest.raises(OSError):
                await asyncio.wait_for(server.run(devmode=True), timeout=10)

        assert fake_executor.started
        started_at = time.monotonic()
        await asyncio.wait_for(server.aclose(), timeout=10)

        assert time.monotonic() - started_at < 5
        assert fake_executor.closed and not fake_executor.is_alive()
        assert server._http_server._runner is None
    finally:
        if not fake_executor.closed:
            await fake_executor.aclose()
        await server._http_server.aclose()
        blocker.close()


# -- health endpoint: not ready until RegisterWorkerResponse


async def test_health_returns_503_before_register_response() -> None:
    register_requests = 0

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        nonlocal register_requests
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:  # never reply to RegisterWorkerRequest
            register_requests += 1
        return ws

    async with _fake_livekit(agent_ws) as port:
        server = _registered_server(port, health_port=18881)
        registered = asyncio.Event()
        server.on("worker_registered", lambda *_: registered.set())
        run_task = asyncio.create_task(server.run(devmode=False))
        try:
            while (health := await _health(18881)) is None or register_requests == 0:
                await asyncio.sleep(0.01)

            assert health[0] == 503 and "not connected" in health[1].lower()
            assert not registered.is_set()
        finally:
            await server.aclose()
            await run_task


async def test_health_ok_when_unregistered() -> None:
    """unregistered=True never connects, so / stays a liveness probe."""
    server = AgentServer(host="127.0.0.1", port=18882, num_idle_processes=0)
    server.rtc_session()(_noop_entrypoint)

    process_pool = MagicMock()
    process_pool.start = AsyncMock()
    process_pool.aclose = AsyncMock()

    with patch("livekit.agents.ipc.proc_pool.ProcPool", return_value=process_pool):
        run_task = asyncio.create_task(server.run(unregistered=True, devmode=True))
        try:
            while (health := await _health(18882)) is None:
                await asyncio.sleep(0.01)
            assert health == (200, "OK")
        finally:
            await server.aclose()
            await run_task


@pytest.mark.parametrize(
    ("grace_period", "expected"), [(30.0, 200), (0.05, 503)], ids=["short_drop", "long_drop"]
)
async def test_health_after_a_registered_connection_drops(
    grace_period: float, expected: int
) -> None:
    """A dropped connection stays healthy for the grace period, then reports 503."""
    connections = 0

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connections += 1
        async for _ in ws:
            if connections == 1:  # register once, then drop; later attempts never register
                reply = agent.ServerMessage()
                reply.register.worker_id = "W_test_reconnect"
                await ws.send_bytes(reply.SerializeToString())
                await ws.close()
        return ws

    with patch("livekit.agents.worker.HEALTH_RECONNECT_GRACE_PERIOD", grace_period):
        async with _fake_livekit(agent_ws) as port:
            server = _registered_server(port, health_port=18884)
            run_task = asyncio.create_task(server.run(devmode=False))
            try:
                while connections < 2:  # the worker noticed the drop and is reconnecting
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.1)
                health = await _health(18884)
            finally:
                await server.aclose()
                await run_task

    assert health is not None and health[0] == expected


# -- job status


async def _status_entrypoint(ctx: JobContext) -> None:
    async def disconnect(
        *, reason: rtc.DisconnectReason.ValueType = rtc.DisconnectReason.CLIENT_INITIATED
    ) -> None:
        async with aiohttp.ClientSession() as http:
            async with http.post(ctx.job.room.metadata, json={"reason": reason}) as response:
                response.raise_for_status()

    # the worker connection and executors are real; only RTC is mocked
    room = create_mock_room()
    room.disconnect.side_effect = disconnect
    room.remote_participants = {}
    scenario = ctx.job.metadata
    if scenario == "raise_before_connect":
        raise RuntimeError("entrypoint failed before connect")

    await ctx.connect()
    if scenario == "raise_after_connect":
        raise RuntimeError("entrypoint failed after connect")

    ctx.shutdown()
    if scenario == "shutdown_then_raise":
        raise RuntimeError("entrypoint failed after shutdown")


_FAILED = (agent.JobStatus.JS_FAILED, rtc.DisconnectReason.AGENT_ERROR)
_SUCCESS = (agent.JobStatus.JS_SUCCESS, rtc.DisconnectReason.CLIENT_INITIATED)


@pytest.mark.parametrize(
    ("executor_type", "scenario", "expected"),
    [
        (JobExecutorType.THREAD, "raise_before_connect", _FAILED),
        (JobExecutorType.THREAD, "raise_after_connect", _FAILED),
        (JobExecutorType.THREAD, "shutdown_then_raise", _SUCCESS),
        (JobExecutorType.THREAD, "normal_shutdown", _SUCCESS),
        (JobExecutorType.PROCESS, "raise_after_connect", _FAILED),
        (JobExecutorType.PROCESS, "normal_shutdown", _SUCCESS),
    ],
)
async def test_final_job_status_reaches_worker_websocket(
    executor_type: JobExecutorType,
    scenario: str,
    expected: tuple[agent.JobStatus, rtc.DisconnectReason.ValueType],
) -> None:
    terminal: asyncio.Future[agent.UpdateJobStatus] = asyncio.get_running_loop().create_future()
    job = agent.Job(
        id="AJ_status_test",
        type=agent.JobType.JT_ROOM,
        room=models.Room(name="status_test"),
        metadata=scenario,
    )
    updates: list[agent.UpdateJobStatus] = []
    disconnect_reasons: list[int] = []

    async def rtc_disconnect(request: web.Request) -> web.Response:
        disconnect_reasons.append((await request.json())["reason"])
        return web.Response()

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for packet in ws:
            if packet.type != aiohttp.WSMsgType.BINARY:
                continue
            message = agent.WorkerMessage.FromString(packet.data)
            if message.HasField("register"):
                register = agent.RegisterWorkerResponse(worker_id="AW_status_test")
                await ws.send_bytes(agent.ServerMessage(register=register).SerializeToString())
                availability = agent.AvailabilityRequest(job=job)
                await ws.send_bytes(
                    agent.ServerMessage(availability=availability).SerializeToString()
                )
            elif message.HasField("availability"):
                assignment = agent.JobAssignment(job=job, token="fake_token")
                await ws.send_bytes(agent.ServerMessage(assignment=assignment).SerializeToString())
            elif message.HasField("update_job"):
                updates.append(message.update_job)
                terminal_states = (agent.JobStatus.JS_FAILED, agent.JobStatus.JS_SUCCESS)
                if message.update_job.status in terminal_states and not terminal.done():
                    terminal.set_result(message.update_job)
        return ws

    async with _fake_livekit(agent_ws, rtc_disconnect=rtc_disconnect) as port:
        job.room.metadata = f"http://127.0.0.1:{port}/rtc_disconnect"
        server = AgentServer(
            ws_url=f"ws://127.0.0.1:{port}",
            api_key="devkey",
            api_secret="secret" * 6,
            job_executor_type=executor_type,
            num_idle_processes=0,
            multiprocessing_context="spawn",
            max_retry=0,
        )
        server.rtc_session()(_status_entrypoint)
        server._simulation = True  # skip the health HTTP server and load checks
        launch_job = ProcPool.launch_job

        async def launch_fake_job(pool: ProcPool, info: RunningJobInfo) -> None:
            info.fake_job = True
            await launch_job(pool, info)

        # create_mock_room() is a process-wide singleton: undo the entrypoint's changes to it
        mock_room = create_mock_room()
        with (
            patch.object(ProcPool, "launch_job", launch_fake_job),
            patch.object(mock_room, "disconnect", new_callable=AsyncMock),
            patch.object(mock_room, "remote_participants", {}),
        ):
            task = asyncio.create_task(server.run(devmode=True))
            try:
                final = await asyncio.wait_for(terminal, timeout=20)
            finally:
                await server.aclose()
                await task

    assert final.job_id == job.id
    assert (final.status, *disconnect_reasons) == expected
    assert updates[0].status == agent.JobStatus.JS_RUNNING
    if scenario != "raise_before_connect":
        assert any(update.join_timing.connected_us > 0 for update in updates)


_REGIONS = [
    {"region": "us-east", "url": "https://proj.us-east.example", "distance": "10"},
    {"region": "us-west", "url": "https://proj.us-west.example", "distance": "4000"},
]


def test_pick_region_url_prefers_the_worker_region() -> None:
    assert _pick_region_url(_REGIONS, "us-west") == "https://proj.us-west.example"


def test_pick_region_url_falls_back_to_closest() -> None:
    assert _pick_region_url(_REGIONS, "eu-central") == "https://proj.us-east.example"
    assert _pick_region_url(_REGIONS, "") == "https://proj.us-east.example"


def test_pick_region_url_without_regions() -> None:
    assert _pick_region_url([], "us-east") is None


def test_is_cloud_url() -> None:
    assert _is_cloud_url("wss://proj.livekit.cloud")
    assert _is_cloud_url("https://proj.staging.livekit.run")
    assert not _is_cloud_url("ws://10.0.0.19:7880")
    assert not _is_cloud_url("wss://livekit.example.com")
