"""Past worker bugs reproduced through real job processes and local services.

Single-job cases run on a job runner (``processes.job_runner``); the in-process server cases
start the server through its private ``_run`` and ``_simulate_job``, the pool and IPC a
dispatched job uses, since an ``AgentServer`` has no public in-process entry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import textwrap
from collections.abc import AsyncIterator
from multiprocessing.queues import Queue
from pathlib import Path

import psutil
import pytest
from aiohttp import web
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from livekit import api
from livekit.agents import Agent, AgentServer, _core
from livekit.agents._generated.worker_options import ProcessOptions
from livekit.agents.ipc import JobLauncher
from livekit.agents.job import JobContext, JobExecutorType
from livekit.protocol import agent, models
from tests.fakes import FakeLLM, tool_call, until

from .processes import OwnRunner, job_runner, running_job
from .resources import child_process

KEY, SECRET = "APIregression", "regression-secret-regression-secret"


def _record(ctx: JobContext, suffix: str, value: object) -> None:
    Path(ctx.job.metadata + suffix).write_text(json.dumps(value))


def _record_shutdown(ctx: JobContext) -> None:
    async def closed(reason: str) -> None:
        _record(ctx, ".closed", reason)

    ctx.add_shutdown_callback(closed)


def _read(marker: Path, suffix: str) -> object:
    return json.loads(marker.with_suffix(suffix).read_text())


def _compute(answers: Queue[dict[str, int]]) -> None:
    answers.put({"pid": os.getpid(), "answer": 42})


async def _tool_job(ctx: JobContext) -> None:
    _record_shutdown(ctx)
    desk = Agent(
        instructions="Use compute.", llm=FakeLLM([tool_call("compute"), "computed", "served"])
    )

    @desk.tool
    async def compute() -> int:
        forkserver = ctx.proc.user_arguments == "forkserver"
        context = mp.get_context("forkserver") if forkserver else mp.get_context("spawn")
        answers: Queue[dict[str, int]] = context.Queue()
        child = context.Process(target=_compute, args=(answers,))
        child.start()
        result = await asyncio.to_thread(answers.get, timeout=8)
        await asyncio.to_thread(child.join, 5)
        _record(ctx, ".child", {**result, "exitcode": child.exitcode})
        return result["answer"]

    first = await desk.run("compute")
    second = await desk.run("continue")
    _record(ctx, ".result", [first.status, second.status])
    ctx.shutdown("tool completed")


async def _crashed_job(ctx: JobContext) -> None:
    _record_shutdown(ctx)
    raise ValueError("regression entrypoint refused")


async def _log_job(ctx: JobContext) -> None:
    logger = logging.getLogger("regression.custom")
    logger.info("regression suppressed info")
    logger.error("regression enabled error")
    ctx.shutdown("logged")


async def _api_job(ctx: JobContext) -> None:
    rooms = await ctx.api.room.list_rooms(api.ListRoomsRequest())
    Path(ctx.job.room.metadata).write_text(json.dumps([room.name for room in rooms.rooms]))
    ctx.shutdown("API completed")


async def _run_one_job(process: _core.JobSupervisor, marker: Path) -> None:
    await process.start()
    await process.initialize()
    await process.launch_job(running_job(agent.Job(id="AJ_regression", metadata=str(marker))))
    await process.join()


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
async def test_a_tool_can_spawn_and_join_a_child_and_the_job_keeps_serving(
    method: str, own_executor: OwnRunner, tmp_path: Path
) -> None:
    """A tool's own child process runs, is joined, and the job answers the next turn.
    https://github.com/livekit/agents/issues/4004"""
    marker = tmp_path / "tool"
    process = own_executor(
        job_runner(_tool_job, user_arguments=method, mp_ctx=mp.get_context("spawn"))
    )

    await _run_one_job(process, marker)

    assert process.status is not None and process.status.value == "success"
    assert _read(marker, ".result") == ["completed", "completed"]
    child = _read(marker, ".child")
    assert isinstance(child, dict) and (child["answer"], child["exitcode"]) == (42, 0)
    assert not psutil.pid_exists(child["pid"]), "the tool's child was not reaped"


@pytest.mark.parametrize("thread", [False, True])
@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8445 reproduces: crashed jobs report success",
)
async def test_an_entrypoint_exception_reports_failed_after_shutdown_callbacks(
    thread: bool, own_executor: OwnRunner, tmp_path: Path
) -> None:
    """A crashed entrypoint runs its shutdown callbacks and reports failure.
    https://github.com/livekit/agents/pull/8445"""
    marker = tmp_path / "crashed"
    process = own_executor(job_runner(_crashed_job, thread=thread, mp_ctx=mp.get_context("spawn")))

    await _run_one_job(process, marker)

    assert _read(marker, ".closed") == "job crashed"
    assert process.status is not None and process.status.value == "failed"
    assert process.exitcode != 0


async def test_a_job_process_honours_the_parents_custom_logger_level(
    own_executor: OwnRunner,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A level the application set on its logger filters that logger in the job process too.
    https://github.com/livekit/agents/pull/4139"""
    monkeypatch.setattr(logging.getLogger("regression.custom"), "level", logging.ERROR)
    process = own_executor(job_runner(_log_job, mp_ctx=mp.get_context("spawn")))

    await _run_one_job(process, tmp_path / "logging")

    output = "".join(capfd.readouterr())
    assert process.exitcode == 0
    assert "regression enabled error" in output
    assert "regression suppressed info" not in output


@pytest.fixture
async def room_service() -> AsyncIterator[tuple[int, list[str]]]:
    """A RoomService on loopback answering ``ListRooms`` for tokens signed with ``KEY``; yields
    its port and the keys each request was signed with."""
    signed_with: list[str] = []

    async def list_rooms(request: web.Request) -> web.Response:
        token = request.headers["Authorization"].removeprefix("Bearer ")
        grants = api.TokenVerifier(KEY, SECRET).verify(token)
        assert grants.video is not None and grants.video.room_list
        signed_with.append(KEY)
        rooms = api.ListRoomsResponse(rooms=[models.Room(name="authenticated")])
        return web.Response(body=rooms.SerializeToString(), content_type="application/protobuf")

    app = web.Application()
    app.router.add_post("/twirp/livekit.RoomService/ListRooms", list_rooms)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield runner.addresses[0][1], signed_with
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("executor", [JobExecutorType.PROCESS, JobExecutorType.THREAD])
@pytest.mark.parametrize("environment", ["absent", "conflicting"])
async def test_the_servers_own_credentials_sign_the_jobs_api_requests(
    executor: JobExecutorType,
    environment: str,
    room_service: tuple[int, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ctx.api`` uses the credentials the server was built with, whatever the environment
    holds. https://github.com/livekit/agents/pull/7929"""
    port, signed_with = room_service
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if environment == "absent":
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "http://127.0.0.1:1" if name == "LIVEKIT_URL" else "wrong")
    monkeypatch.setenv("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")
    marker = tmp_path / "rooms.json"
    server = AgentServer(
        ws_url=f"ws://127.0.0.1:{port}",
        api_key=KEY,
        api_secret=SECRET,
        num_idle_processes=0,
        listeners={"health": {"port": 0}},
    )
    server._update_options(ProcessOptions(job_executor_type=executor))
    server.rtc_session()(_api_job)
    ready = asyncio.Event()
    server._on_started.append(ready.set)
    run = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        await ready.wait()
        room = models.Room(name="credentials", metadata=str(marker))
        await server._simulate_job("credentials", fake_job=True, room_info=room)
        await until(marker.exists, "the job listed no rooms", 15)
    finally:
        await server.aclose()
        await run

    assert signed_with == [KEY]
    assert json.loads(marker.read_text()) == ["authenticated"]


STUB_SERVER = """
import asyncio
from pathlib import Path
from livekit.agents import AgentServer, cli

class Server(AgentServer):
    async def _run(self, **kwargs):
        Path(__file__).with_suffix(".ready").touch()
        await asyncio.Event().wait()
    async def _drain(self, timeout=None):
        Path(__file__).with_suffix(".drained").touch()
    async def aclose(self):
        Path(__file__).with_suffix(".closed").touch()

cli.run_app(Server(shutdown_timeout=3, drain_timeout=1))
"""


@pytest.mark.parametrize("development", [False, True], ids=["start", "start_dev"])
async def test_sigterm_drains_in_production_and_skips_the_drain_in_development(
    development: bool, tmp_path: Path
) -> None:
    """SIGTERM closes the server in both modes; only production drains jobs first.
    https://github.com/livekit/agents/pull/4866"""
    script = tmp_path / "agent.py"
    script.write_text(textwrap.dedent(STUB_SERVER))
    args = [sys.executable, str(script), "start", *(["--dev"] if development else [])]
    async with child_process(*args, log=tmp_path / "launcher.log") as child:
        await until(script.with_suffix(".ready").exists, child.log.read_text(), 15)

        child.process.send_signal(signal.SIGTERM)

        assert await child.process.wait() == 0, child.log.read_text()
    assert script.with_suffix(".closed").exists()
    assert script.with_suffix(".drained").exists() is not development


async def test_a_failed_process_creation_releases_both_ipc_sockets(
    own_executor: OwnRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch failing before the OS created the process leaks no file descriptor.
    https://github.com/livekit/agents/pull/4823"""

    def fail_creation(self: JobLauncher, context: object, endpoint: object) -> object:
        raise OSError("regression could not create process")

    monkeypatch.setattr(JobLauncher, "_create_process", fail_creation)

    async def fail_once() -> None:
        process = own_executor(job_runner(_log_job, mp_ctx=mp.get_context("spawn")))
        with pytest.raises(OSError, match="regression could not create process"):
            await process.start()
        await process.aclose()

    await fail_once()  # the first failure starts the process-wide supervisor services
    baseline = psutil.Process().num_fds()

    for _ in range(5):
        await fail_once()

    assert psutil.Process().num_fds() <= baseline


@pytest.fixture
async def otlp_collector() -> AsyncIterator[tuple[str, list[str]]]:
    """An OTLP/HTTP trace collector on loopback, as Cloud observability serves it; yields its
    base URL and the name of every span it received."""
    spans: list[str] = []

    async def traces(request: web.Request) -> web.Response:
        export = ExportTraceServiceRequest.FromString(await request.read())
        for resource in export.resource_spans:
            for scope in resource.scope_spans:
                spans.extend(span.name for span in scope.spans)
        return web.Response(body=b"", content_type="application/x-protobuf")

    app = web.Application()
    app.router.add_post("/observability/traces/otlp/v0", traces)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}", spans
    finally:
        await runner.cleanup()


async def _recorded_job_that_crashes(ctx: JobContext) -> None:
    ctx.init_recording({"audio": False, "traces": True, "logs": False, "transcript": False})
    raise ValueError("regression entrypoint refused")


async def test_a_crashed_jobs_spans_reach_the_observability_url_before_it_exits(
    own_executor: OwnRunner,
    otlp_collector: tuple[str, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded job whose entrypoint raises still flushes its trace to
    `LIVEKIT_OBSERVABILITY_URL` before its process exits. https://github.com/livekit/agents/pull/4082"""
    url, spans = otlp_collector
    monkeypatch.setenv("LIVEKIT_OBSERVABILITY_URL", url)
    credentials = (("LIVEKIT_API_KEY", KEY), ("LIVEKIT_API_SECRET", SECRET))
    process = own_executor(
        job_runner(_recorded_job_that_crashes, env=credentials, mp_ctx=mp.get_context("spawn"))
    )
    job = agent.Job(id="AJ_crashed", room=models.Room(name="crashed"), metadata=str(tmp_path / "c"))

    await process.start()
    await process.initialize()
    await process.launch_job(running_job(job, fake_job=False))
    await process.join()

    assert "job_entrypoint" in spans
