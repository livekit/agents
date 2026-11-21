from __future__ import annotations

import asyncio
import contextlib
import json
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

from livekit import api, rtc
from livekit.agents import AgentServer, JobContext, JobExecutorType
from livekit.agents.cli import proto
from livekit.agents.cli.cli import _run_worker
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.mock_room import create_mock_room
from livekit.agents.ipc.proc_pool import ProcPool
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
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


@pytest.mark.virtual_time
async def test_worker_times_out_waiting_for_registration() -> None:
    await asyncio.sleep(0.001)  # start the worker's intervals after virtual time zero
    server = AgentServer(
        ws_url="ws://localhost",
        api_key="devkey",
        api_secret="secret" * 6,
        num_idle_processes=0,
        max_retry=0,
    )
    server.rtc_session()(_noop_entrypoint)
    server._simulation = True
    ws = AsyncMock(spec=aiohttp.ClientWebSocketResponse)

    async def wait_for_response() -> None:
        await asyncio.sleep(60)

    ws.receive_bytes.side_effect = wait_for_response
    with (
        patch.dict(os.environ),
        patch.object(aiohttp.ClientSession, "ws_connect", new_callable=AsyncMock, return_value=ws),
    ):
        try:
            with pytest.raises(RuntimeError, match="failed to connect to livekit"):
                await asyncio.wait_for(server.run(devmode=True), timeout=11)
            ws.close.assert_awaited_once()
        finally:
            await server.aclose()


@pytest.mark.virtual_time
async def test_worker_reconnects_when_a_ping_goes_unanswered() -> None:
    await asyncio.sleep(0.001)  # start the worker's intervals after virtual time zero
    server = AgentServer(
        ws_url="ws://localhost", api_key="devkey", api_secret="secret" * 6, num_idle_processes=0
    )
    server.rtc_session()(_noop_entrypoint)
    server._simulation = True
    clock = asyncio.get_running_loop().time
    first_pings: list[float] = []
    recovered_ping = asyncio.Event()
    close_started = 0.0

    def make_socket(worker_id: str) -> AsyncMock:
        incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        ws = AsyncMock(spec=aiohttp.ClientWebSocketResponse)
        ws.receive_bytes.return_value = agent.ServerMessage(
            register=agent.RegisterWorkerResponse(worker_id=worker_id)
        ).SerializeToString()
        ws.receive.side_effect = incoming.get

        async def send(data: bytes) -> None:
            msg = agent.WorkerMessage.FromString(data)
            if not msg.HasField("ping"):
                return
            if worker_id == "AW_first":
                first_pings.append(clock())
                if len(first_pings) > 1:
                    return
            else:
                recovered_ping.set()
            pong = agent.ServerMessage(pong=agent.WorkerPong(last_timestamp=msg.ping.timestamp))
            incoming.put_nowait(
                aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, pong.SerializeToString(), None)
            )

        ws.send_bytes.side_effect = send
        return ws

    async def stall_close() -> None:
        nonlocal close_started
        close_started = clock()
        await asyncio.sleep(10)

    first_ws = make_socket("AW_first")
    first_ws.close.side_effect = stall_close
    with (
        patch.dict(os.environ),
        patch.object(aiohttp.ClientSession, "ws_connect", new_callable=AsyncMock) as connect,
    ):
        connect.side_effect = [first_ws, make_socket("AW_next")]
        run = asyncio.create_task(server.run(devmode=True))
        try:
            await asyncio.wait_for(recovered_ping.wait(), timeout=5)
            assert connect.await_count == 2
            assert server.id == "AW_next"
            first, second = first_pings
            assert second - first == pytest.approx(1.0)
            assert close_started - second == pytest.approx(2.0)
            first_ws.close.assert_awaited_once()
        finally:
            await server.aclose()
            await run


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
            elif message.HasField("ping"):
                pong = agent.WorkerPong(last_timestamp=message.ping.timestamp)
                await ws.send_bytes(agent.ServerMessage(pong=pong).SerializeToString())
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


# -- token mode: LIVEKIT_TOKEN without an API key and secret

_WORKER_TOKEN = "worker-token-without-secret"
_NO_KEY_ENV = {"LIVEKIT_API_KEY": "", "LIVEKIT_API_SECRET": "", "LIVEKIT_TOKEN": _WORKER_TOKEN}


@pytest.mark.parametrize("token_mode", [True, False], ids=["token_mode", "key_mode"])
async def test_registration_bearer(token_mode: bool) -> None:
    """Token mode registers with the worker token as is; with a key and secret, the worker
    signs its own token and ignores LIVEKIT_TOKEN. The hosted worker_token is sent in both."""
    registered: asyncio.Future[web.Request] = asyncio.get_running_loop().create_future()

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            reply = agent.ServerMessage(register=_register_response("AW_t", agent_protocol=2))
            await ws.send_bytes(reply.SerializeToString())
            if not registered.done():
                registered.set_result(request)
        return ws

    env = {**_NO_KEY_ENV, "LIVEKIT_WORKER_TOKEN": "hosted-identity"}
    key = {"api_key": "devkey", "api_secret": "secret" * 6} if not token_mode else {}
    async with _fake_livekit(agent_ws) as port:
        with patch.dict(os.environ, env):
            server = AgentServer(ws_url=f"ws://127.0.0.1:{port}", num_idle_processes=0, **key)
            server.rtc_session()(_noop_entrypoint)
            server._simulation = True  # skip the health HTTP server
            task = asyncio.create_task(server.run(devmode=True))
            try:
                request = await asyncio.wait_for(registered, timeout=10)
            finally:
                await server.aclose()
                await task

    bearer = request.headers["Authorization"].removeprefix("Bearer ")
    assert request.query["worker_token"] == "hosted-identity"
    if token_mode:
        assert bearer == _WORKER_TOKEN
    else:
        claims = api.TokenVerifier("devkey", "secret" * 6).verify(bearer)
        assert claims.video is not None and claims.video.agent


def test_token_mode_needs_no_key_or_secret() -> None:
    with patch.dict(os.environ, _NO_KEY_ENV):
        assert AgentServer()._token_mode
        assert not AgentServer(api_key="devkey", api_secret="secret")._token_mode
        assert not AgentServer(api_key="devkey")._token_mode, "a key without a secret is an error"
        assert AgentServer(token="explicit")._token == "explicit"


def _register_response(worker_id: str, *, agent_protocol: int) -> agent.RegisterWorkerResponse:
    return agent.RegisterWorkerResponse(
        worker_id=worker_id, server_info=models.ServerInfo(agent_protocol=agent_protocol)
    )


def _token_mode_server() -> AgentServer:
    """A token-mode server that is registered, without a connection."""
    with patch.dict(os.environ, _NO_KEY_ENV):
        server = AgentServer(num_idle_processes=0)
    assert server._token_mode
    server._job_token_requests = {}
    server._registered.set()
    return server


@pytest.mark.parametrize(
    ("token_mode", "agent_protocol", "registers"),
    [(True, 1, False), (True, 2, True), (False, 1, True)],
    ids=["token_mode_old_server", "token_mode", "key_mode_old_server"],
)
async def test_token_mode_needs_agent_protocol_2(
    token_mode: bool, agent_protocol: int, registers: bool
) -> None:
    """A server before agent protocol 2 closes the connection on CreateJobToken: token mode
    fails at registration, once, without sending it. Key mode does not care."""
    received: list[str] = []

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for packet in ws:
            message = agent.WorkerMessage.FromString(packet.data)
            received.append(message.WhichOneof("message") or "")
            if message.HasField("register"):
                reply = _register_response("AW_p", agent_protocol=agent_protocol)
                await ws.send_bytes(agent.ServerMessage(register=reply).SerializeToString())
        return ws

    key = {"api_key": "devkey", "api_secret": "secret" * 6} if not token_mode else {}
    async with _fake_livekit(agent_ws) as port:
        with patch.dict(os.environ, _NO_KEY_ENV):
            server = AgentServer(
                ws_url=f"ws://127.0.0.1:{port}", num_idle_processes=0, max_retry=3, **key
            )
            server.rtc_session()(_noop_entrypoint)
            server._simulation = True  # skip the health HTTP server
            registered = asyncio.Event()
            server.on("worker_registered", lambda *_: registered.set())
            task = asyncio.create_task(server.run(devmode=True))
            try:
                if registers:
                    await asyncio.wait_for(registered.wait(), timeout=10)
                else:
                    with pytest.raises(RuntimeError, match="does not support token mode") as exc:
                        await asyncio.wait_for(task, timeout=10)
                    assert "LIVEKIT_API_KEY and LIVEKIT_API_SECRET" in str(exc.value)
                    assert not registered.is_set()
                    with pytest.raises(ConnectionError, match="not connected"):
                        await server._create_job_token("AJ_job", "{}")
            finally:
                await server.aclose()
                with contextlib.suppress(RuntimeError):
                    await task

    assert received.count("register") == 1, "one attempt"
    assert "create_job_token" not in received


async def test_create_job_token_fails_when_the_connection_is_lost() -> None:
    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for packet in ws:
            message = agent.WorkerMessage.FromString(packet.data)
            if message.HasField("register"):
                reply = _register_response("AW_lost", agent_protocol=2)
                await ws.send_bytes(agent.ServerMessage(register=reply).SerializeToString())
            elif message.HasField("create_job_token"):
                await ws.close()  # drop the connection instead of answering
        return ws

    async with _fake_livekit(agent_ws) as port:
        with patch.dict(os.environ, _NO_KEY_ENV):
            server = AgentServer(
                ws_url=f"ws://127.0.0.1:{port}", num_idle_processes=0, max_retry=50
            )
            server.rtc_session()(_noop_entrypoint)
            server._simulation = True  # skip the health HTTP server
            registered = asyncio.Event()
            server.on("worker_registered", lambda *_: registered.set())
            task = asyncio.create_task(server.run(devmode=True))
            try:
                await asyncio.wait_for(registered.wait(), timeout=10)
                with pytest.raises(ConnectionError, match="connection to the LiveKit server"):
                    await asyncio.wait_for(server._create_job_token("AJ_job", "{}"), timeout=5)
            finally:
                await server.aclose()
                await task


async def test_jobs_get_tokens_while_the_worker_closes() -> None:
    """A job's shutdown callbacks call the API, so the connection outlives the jobs."""

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for packet in ws:
            message = agent.WorkerMessage.FromString(packet.data)
            if message.HasField("register"):
                reply = _register_response("AW_closing", agent_protocol=2)
                await ws.send_bytes(agent.ServerMessage(register=reply).SerializeToString())
            elif message.HasField("create_job_token"):
                res = agent.CreateJobTokenResponse(
                    request_id=message.create_job_token.request_id, token="job-token"
                )
                await ws.send_bytes(agent.ServerMessage(create_job_token=res).SerializeToString())
        return ws

    tokens: list[str] = []
    close_pool = ProcPool.aclose

    async def close_jobs(pool: ProcPool) -> None:
        # what a job's shutdown callback asks for through the pool
        tokens.append((await server._create_job_token("AJ_job", "{}")).token)
        await close_pool(pool)

    async with _fake_livekit(agent_ws) as port:
        with patch.dict(os.environ, _NO_KEY_ENV):
            server = AgentServer(ws_url=f"ws://127.0.0.1:{port}", num_idle_processes=0)
            server.rtc_session()(_noop_entrypoint)
            server._simulation = True  # skip the health HTTP server
            registered = asyncio.Event()
            server.on("worker_registered", lambda *_: registered.set())
            task = asyncio.create_task(server.run(devmode=True))
            try:
                await asyncio.wait_for(registered.wait(), timeout=10)
            finally:
                with patch.object(ProcPool, "aclose", close_jobs):
                    await asyncio.wait_for(server.aclose(), timeout=10)
                await task

    assert tokens == ["job-token"]
    assert not server._is_available(), "a closing worker takes no new job"


@pytest.mark.virtual_time
async def test_create_job_token_times_out_when_the_server_never_answers() -> None:
    server = _token_mode_server()
    with patch.object(server, "_queue_msg", new_callable=AsyncMock) as queue_msg:
        with pytest.raises(RuntimeError, match="did not answer the job token request within"):
            await server._create_job_token("AJ_job", "{}")

    assert queue_msg.call_args.args[0].create_job_token.job_id == "AJ_job"
    assert not server._job_token_requests


async def _answer_job_tokens(
    server: AgentServer, sent: list[agent.WorkerMessage], *, error: str = ""
) -> None:
    """A _queue_msg stand-in: records what the worker sends and answers CreateJobToken."""

    async def queue_msg(msg: agent.WorkerMessage) -> None:
        sent.append(msg)
        if msg.HasField("create_job_token"):
            req = msg.create_job_token
            server._job_token_requests[req.request_id].set_result(
                agent.CreateJobTokenResponse(
                    request_id=req.request_id,
                    token="" if error else f"job-token-for-{req.job_id}",
                    expires_at=int(time.time()) + 600,
                    error=error,
                )
            )

    server._queue_msg = queue_msg  # type: ignore[method-assign]


async def test_create_job_token_raises_the_server_error() -> None:
    server = _token_mode_server()
    await _answer_job_tokens(server, [], error="claim not allowed: video.agent")
    with pytest.raises(RuntimeError, match="refused the job token: claim not allowed: video.agent"):
        await server._create_job_token("AJ_job", '{"video": {"agent": true}}')


async def test_hot_reload_asks_for_a_fresh_token_without_video_agent() -> None:
    server = _token_mode_server()
    sent: list[agent.WorkerMessage] = []
    await _answer_job_tokens(server, sent)
    server._proc_pool = MagicMock(launch_job=AsyncMock())

    join_token = (
        api.AccessToken("devkey", "secret" * 6)
        .with_identity("agent-identity")
        .with_kind("agent")
        .with_attributes({"lk.agent.state": "listening"})
        .with_grants(api.VideoGrants(room_join=True, room="reload-room", agent=True))
        .to_jwt()
    )
    running = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent-identity", metadata=""),
        job=agent.Job(id="AJ_reload", room=models.Room(name="reload-room")),
        url="ws://localhost",
        token=join_token,
        worker_id="AW_test",
        fake_job=False,
    )
    with patch.dict(os.environ, _NO_KEY_ENV):
        await server._reload_jobs([running])

    # the server learns the job runs here, from the full job, before it is asked for the
    # job's token; the identity is the one the worker accepted the job with
    assert [m.WhichOneof("message") for m in sent] == ["migrate_job", "create_job_token"]
    assert list(sent[0].migrate_job.job_ids) == ["AJ_reload"]
    migrated = sent[0].migrate_job.jobs
    assert [j.id for j in migrated] == ["AJ_reload"]
    assert migrated[0].room.name == "reload-room"
    assert migrated[0].state.participant_identity == "agent-identity"
    claims = json.loads(sent[1].create_job_token.claims)
    assert sent[1].create_job_token.job_id == "AJ_reload"
    assert claims["identity"] == "agent-identity"
    assert claims["kind"] == "agent"
    assert claims["attributes"] == {"lk.agent.state": "listening"}
    assert claims["video"]["roomJoin"] and claims["video"]["room"] == "reload-room"
    assert "agent" not in claims["video"]

    launched: RunningJobInfo = server._proc_pool.launch_job.call_args.args[0]
    assert launched.token == "job-token-for-AJ_reload"


async def test_hot_reload_skips_a_job_the_server_has_no_token_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A self-hosted server does not keep the job across the restart of its worker."""
    server = _token_mode_server()
    await _answer_job_tokens(server, [], error="job is not running on this worker")
    server._proc_pool = MagicMock(launch_job=AsyncMock())

    running = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent-identity", metadata=""),
        job=agent.Job(id="AJ_reload", room=models.Room(name="reload-room")),
        url="ws://localhost",
        token=api.AccessToken("devkey", "secret" * 6).with_identity("agent-identity").to_jwt(),
        worker_id="AW_test",
        fake_job=False,
    )
    with patch.dict(os.environ, _NO_KEY_ENV), caplog.at_level("WARNING"):
        await server._reload_jobs([running])

    server._proc_pool.launch_job.assert_not_called()
    assert "reload is only supported when LiveKit API key and secret are defined" in caplog.text


async def test_hot_reload_reports_a_job_token_request_that_timed_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A request the server did not answer is not a server without reload support."""
    server = _token_mode_server()
    server._queue_msg = AsyncMock()  # type: ignore[method-assign]  # the server never answers
    server._proc_pool = MagicMock(launch_job=AsyncMock())

    running = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent-identity", metadata=""),
        job=agent.Job(id="AJ_reload", room=models.Room(name="reload-room")),
        url="ws://localhost",
        token=api.AccessToken("devkey", "secret" * 6).with_identity("agent-identity").to_jwt(),
        worker_id="AW_test",
        fake_job=False,
    )
    with (
        patch.dict(os.environ, _NO_KEY_ENV),
        patch("livekit.agents.worker.CREATE_JOB_TOKEN_TIMEOUT", 0.01),
        caplog.at_level("WARNING"),
    ):
        await server._reload_jobs([running])

    server._proc_pool.launch_job.assert_not_called()
    assert "job ended on reload: the LiveKit server did not answer" in caplog.text
    assert "reload is only supported" not in caplog.text


async def _token_mode_api_entrypoint(ctx: JobContext) -> None:
    await ctx.api.room.list_participants(api.ListParticipantsRequest(room=ctx.job.room.name))
    ctx.shutdown()


async def test_ctx_api_gets_its_token_from_create_job_token() -> None:
    """End to end on a fake server: a job's ctx.api call in token mode asks the server for a
    token with the call's grants, through the worker, and sends that token."""
    job = agent.Job(id="AJ_api", type=agent.JobType.JT_ROOM, room=models.Room(name="api-room"))
    token_claims: list[dict[str, Any]] = []
    api_call: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def list_participants(request: web.Request) -> web.Response:
        if not api_call.done():
            api_call.set_result(request.headers["Authorization"])
        body = api.ListParticipantsResponse().SerializeToString()
        return web.Response(body=body, content_type="application/protobuf")

    async def agent_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for packet in ws:
            if packet.type != aiohttp.WSMsgType.BINARY:
                continue
            message = agent.WorkerMessage.FromString(packet.data)
            if message.HasField("register"):
                register = _register_response("AW_api", agent_protocol=2)
                await ws.send_bytes(agent.ServerMessage(register=register).SerializeToString())
                availability = agent.AvailabilityRequest(job=job)
                await ws.send_bytes(
                    agent.ServerMessage(availability=availability).SerializeToString()
                )
            elif message.HasField("availability"):
                assignment = agent.JobAssignment(job=job, token="join-token")
                await ws.send_bytes(agent.ServerMessage(assignment=assignment).SerializeToString())
            elif message.HasField("create_job_token"):
                req = message.create_job_token
                assert req.job_id == job.id
                token_claims.append(json.loads(req.claims))
                res = agent.CreateJobTokenResponse(
                    request_id=req.request_id,
                    token=f"job-token-{len(token_claims)}",
                    expires_at=int(time.time()) + 600,
                )
                await ws.send_bytes(agent.ServerMessage(create_job_token=res).SerializeToString())
        return ws

    routes = {"twirp/livekit.RoomService/ListParticipants": list_participants}
    async with _fake_livekit(agent_ws, **routes) as port:
        with patch.dict(os.environ, _NO_KEY_ENV):
            server = AgentServer(
                ws_url=f"ws://127.0.0.1:{port}",
                job_executor_type=JobExecutorType.THREAD,
                num_idle_processes=0,
                max_retry=0,
            )
            server.rtc_session()(_token_mode_api_entrypoint)
            server._simulation = True  # skip the health HTTP server and load checks
            launch_job = ProcPool.launch_job

            async def launch_fake_job(pool: ProcPool, info: RunningJobInfo) -> None:
                info.fake_job = True
                await launch_job(pool, info)

            with patch.object(ProcPool, "launch_job", launch_fake_job):
                task = asyncio.create_task(server.run(devmode=True))
                try:
                    authorization = await asyncio.wait_for(api_call, timeout=20)
                finally:
                    await server.aclose()
                    await task

    assert authorization == "Bearer job-token-1"
    assert token_claims == [
        {
            "video": {
                "roomAdmin": True,
                "room": "api-room",
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
            }
        }
    ]


async def test_hot_reload_waits_until_registered() -> None:
    server = _token_mode_server()
    server._registered.clear()
    sent: list[agent.WorkerMessage] = []
    await _answer_job_tokens(server, sent)
    server._proc_pool = MagicMock(launch_job=AsyncMock())
    running = RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="agent-identity", metadata=""),
        job=agent.Job(id="AJ_reload"),
        url="ws://localhost",
        token=api.AccessToken("devkey", "secret" * 6).with_identity("agent-identity").to_jwt(),
        worker_id="AW_test",
        fake_job=False,
    )
    with patch.dict(os.environ, _NO_KEY_ENV):
        reload_task = asyncio.create_task(server._reload_jobs([running]))
        await asyncio.sleep(0.01)
        assert sent == [], "nothing is sent before the worker is registered"

        server._registered.set()
        await asyncio.wait_for(reload_task, timeout=5)

    assert [m.WhichOneof("message") for m in sent] == ["migrate_job", "create_job_token"]


async def test_update_token_replaces_the_worker_token() -> None:
    """A refreshed worker token is used by the next connection. Jobs do not use it: they
    get their own tokens from the server."""
    from livekit.agents._token_mode import token_mode_token

    server = _token_mode_server()
    with patch.dict(os.environ, _NO_KEY_ENV):
        # the dev channel can deliver one before the server runs
        await server._update_token("early-worker-token")
        assert token_mode_token() == "early-worker-token"
        await server._update_token("refreshed-worker-token")
        assert token_mode_token() == "refreshed-worker-token"

    assert server._token == "refreshed-worker-token"


async def test_watch_client_reports_token_mode_and_applies_credentials() -> None:
    """The agent tells the CLI it supports token mode, and applies Credentials whether they
    come before or after the reply to its GetRunningJobsRequest."""
    from livekit.agents.cli.watcher import WatchClient, _recv_proto, _send_proto
    from livekit.protocol.agent_pb import agent_dev

    tokens: list[str] = []
    both_applied = asyncio.Event()

    async def update_token(token: str) -> None:
        tokens.append(token)
        if len(tokens) == 2:
            both_applied.set()

    server = MagicMock(
        _agent_name="my-agent",
        _ws_url="wss://example.livekit.cloud",
        _update_token=update_token,
        _reload_jobs=AsyncMock(),
    )
    received: list[agent_dev.AgentDevMessage] = []
    done = asyncio.Event()

    async def cli(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        for _ in range(2):  # ServerInfo, GetRunningJobsRequest
            received.append(agent_dev.AgentDevMessage.FromString(await _recv_proto(reader)))
        for token in ("first", "second"):
            creds = agent_dev.Credentials(url="wss://x", token=token, expires_at=1)
            await _send_proto(
                writer, agent_dev.AgentDevMessage(credentials=creds).SerializeToString()
            )
            if token == "first":
                jobs = agent_dev.GetRunningAgentJobsResponse()
                msg = agent_dev.AgentDevMessage(get_running_jobs_response=jobs)
                await _send_proto(writer, msg.SerializeToString())
        await done.wait()
        writer.close()

    cli_server = await asyncio.start_server(cli, "127.0.0.1", 0)
    port = cli_server.sockets[0].getsockname()[1]
    client = WatchClient(server, f"127.0.0.1:{port}", loop=asyncio.get_running_loop())
    client.start()
    try:
        await asyncio.wait_for(both_applied.wait(), timeout=5)
    finally:
        done.set()
        await client.aclose()
        cli_server.close()
        await cli_server.wait_closed()

    assert received[0].server_info.supports_token_mode
    assert received[0].server_info.agent_name == "my-agent"
    assert received[1].HasField("get_running_jobs_request")
    assert tokens == ["first", "second"]
    server._reload_jobs.assert_not_awaited()  # no jobs to reload


async def test_watch_client_applies_credentials_while_jobs_reload() -> None:
    """A job reload that waits for registration does not block the Credentials that come
    after it, which registration can need."""
    from livekit.agents.cli.watcher import WatchClient, _recv_proto, _send_proto
    from livekit.protocol.agent_pb import agent_dev

    token_applied = asyncio.Event()
    reloaded: list[list[RunningJobInfo]] = []

    async def update_token(token: str) -> None:
        token_applied.set()

    async def reload_jobs(jobs: list[RunningJobInfo]) -> None:
        await token_applied.wait()  # registers only with the token that comes next
        reloaded.append(jobs)

    server = MagicMock(
        _agent_name="my-agent",
        _ws_url="wss://example.livekit.cloud",
        _update_token=update_token,
        _reload_jobs=reload_jobs,
    )
    done = asyncio.Event()

    async def cli(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        for _ in range(2):  # ServerInfo, GetRunningJobsRequest
            await _recv_proto(reader)
        job = agent_dev.RunningAgentJobInfo(job=agent.Job(id="job-1").SerializeToString())
        jobs = agent_dev.GetRunningAgentJobsResponse(jobs=[job])
        msg = agent_dev.AgentDevMessage(get_running_jobs_response=jobs)
        await _send_proto(writer, msg.SerializeToString())
        creds = agent_dev.Credentials(url="wss://x", token="fresh", expires_at=1)
        await _send_proto(writer, agent_dev.AgentDevMessage(credentials=creds).SerializeToString())
        await done.wait()
        writer.close()

    cli_server = await asyncio.start_server(cli, "127.0.0.1", 0)
    port = cli_server.sockets[0].getsockname()[1]
    client = WatchClient(server, f"127.0.0.1:{port}", loop=asyncio.get_running_loop())
    client.start()
    try:
        await asyncio.wait_for(token_applied.wait(), timeout=5)
        assert client._reload_task is not None
        await asyncio.wait_for(client._reload_task, timeout=5)
    finally:
        done.set()
        await client.aclose()
        cli_server.close()
        await cli_server.wait_closed()

    assert [[j.job.id for j in jobs] for jobs in reloaded] == [["job-1"]]
