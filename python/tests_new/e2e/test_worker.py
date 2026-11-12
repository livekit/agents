"""Job processes on the machine's own processes and sockets, which the DST does not model: a job
runner's launch, health kill and reaping, the warm pool, resource release across many jobs, and a
real worker's supervision of crashed, leaking, slow and orphaned jobs.

The runner tests drive ``_core.JobSupervisor`` over a ``JobLauncher`` (``processes.job_runner``):
the worker's own job runner, which has no public entry of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil
import pytest

from livekit.agents import JobContext, _core
from livekit.agents.ipc import JobLauncher, JobRunnerOptions
from livekit.protocol import agent
from tests_new.fakes import until

from .drivers import Workers
from .processes import OwnRunner, job_runner, process_snapshot, running_job
from .resources import StartJobWorker, gone

METHODS = ["spawn", *(["forkserver"] if sys.platform == "linux" else [])]
PING = 0.05
PING_TIMEOUT = 0.2


async def _finishes(ctx: JobContext) -> None:
    ctx.shutdown("done")


async def _blocks_its_loop(ctx: JobContext) -> None:
    Path(ctx.job.metadata).touch()
    time.sleep(60)  # the job process's event loop answers no ping while this runs


def _assert_reaped(process: _core.JobSupervisor) -> None:
    """The runner's process exited and was waited for: a zombie still exists."""
    assert process.exitcode is not None
    assert process.pid is not None and not psutil.pid_exists(process.pid)


class GatedLauncher(JobLauncher):
    """A launcher whose OS launch waits for ``launch_gate``, then reports ``spawned``, and whose
    launch result is held until ``result_gate``."""

    def __init__(self, method: str) -> None:
        super().__init__(
            job_entrypoint_fnc=_finishes,
            session_end_fnc=None,
            simulation_end_fnc=None,
            session_end_timeout=0.25,
            mp_ctx=mp.get_context(method),
            loop=asyncio.get_running_loop(),
        )
        self.entered = asyncio.Event()
        self.launch_gate = asyncio.Event()
        self.spawned = asyncio.Event()
        self.result_gate = asyncio.Event()

    async def _core_launch(
        self, endpoint: int, launch: _core.ProcessLaunch | _core.ThreadLaunch
    ) -> None:
        self.entered.set()
        await self.launch_gate.wait()
        await super()._core_launch(endpoint, launch)
        self.spawned.set()
        await self.result_gate.wait()

    def release(self) -> None:
        self.launch_gate.set()
        self.result_gate.set()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cancelled", ["before_spawn", "after_spawn"])
async def test_a_cancelled_start_still_reaps_its_process(
    method: str, cancelled: str, own_executor: OwnRunner
) -> None:
    """Cancelling ``start()`` before or after the OS created the process leaves nothing behind."""
    launcher = GatedLauncher(method)
    process = own_executor(
        _core.JobSupervisor(launcher, JobRunnerOptions(initialize_timeout=10, close_timeout=0.25))
    )
    starting = asyncio.ensure_future(process.start())
    await launcher.entered.wait()
    if cancelled == "after_spawn":
        launcher.launch_gate.set()
        await launcher.spawned.wait()

    starting.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await starting
    launcher.launch_gate.set()
    if cancelled == "before_spawn":
        launcher.result_gate.set()
    await process.aclose()  # after a spawn, the close reaps while the launch result is still held
    await process.join()

    _assert_reaped(process)
    launcher.release()


@pytest.mark.parametrize("method", METHODS)
async def test_a_job_process_whose_loop_is_blocked_is_killed_by_its_health_check(
    method: str, own_executor: OwnRunner, tmp_path: Path
) -> None:
    """A job blocking its event loop misses its pings and is killed within the ping timeout."""
    entered = tmp_path / "entered"
    process = own_executor(
        job_runner(
            _blocks_its_loop,
            mp_ctx=mp.get_context(method),
            ping_interval=PING,
            ping_timeout=PING_TIMEOUT,
        )
    )
    await process.start()
    await process.initialize()

    started = time.monotonic()
    with contextlib.suppress(RuntimeError, ConnectionError):
        await process.launch_job(running_job(agent.Job(id="AJ_blocked", metadata=str(entered))))
    await process.join()

    assert entered.exists()
    assert process.killed
    assert time.monotonic() - started < 3, "killed long after the 0.2 s ping timeout"
    _assert_reaped(process)


@pytest.mark.parametrize("method", METHODS)
async def test_a_process_whose_launch_fails_after_spawn_is_still_reaped(
    method: str, own_executor: OwnRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch failing once the OS created the process does not orphan that process."""
    module = "forkserver" if method == "forkserver" else "spawn_posix"
    popen = importlib.import_module(f"multiprocessing.popen_{module}").Popen
    spawn = popen.__init__

    def fail_after_spawn(self: object, process: object) -> None:
        spawn(self, process)
        raise OSError("bootstrap failed after the OS launch")

    monkeypatch.setattr(popen, "__init__", fail_after_spawn)
    process = own_executor(job_runner(_finishes, mp_ctx=mp.get_context(method)))

    await process.start()
    assert process.pid is not None
    await process.aclose()
    await process.join()

    _assert_reaped(process)
    assert process.exitcode != 0


@pytest.mark.parametrize("method", METHODS)
async def test_the_pool_reaps_a_killed_idle_process_and_replaces_it(method: str) -> None:
    """An idle pool process killed by the OS is reaped and replaced, even while the host loop is
    busy."""

    async def launcher() -> JobLauncher:
        return JobLauncher(
            job_entrypoint_fnc=_finishes,
            session_end_fnc=None,
            simulation_end_fnc=None,
            session_end_timeout=0.25,
            mp_ctx=mp.get_context(method),
            loop=asyncio.get_running_loop(),
        )

    options = JobRunnerOptions(
        initialize_timeout=10, close_timeout=0.25, ping_interval=PING, ping_timeout=PING_TIMEOUT
    )
    pool = _core.ProcPool(launcher, num_idle_processes=1, options=options)
    try:
        await pool.start()
        [killed] = pool.processes
        assert killed.pid is not None

        os.kill(killed.pid, signal.SIGKILL)
        time.sleep(1)  # the host loop is blocked: reaping must not depend on it
        await killed.join()

        _assert_reaped(killed)

        def replacement() -> list[_core.JobSupervisor]:
            return [p for p in pool.processes if p.id != killed.id and p.initialized]

        await until(lambda: replacement(), "no replacement process became ready", 15)
        assert killed.id not in {p.id for p in pool.processes}
    finally:
        await pool.aclose()


@pytest.mark.parametrize("method", METHODS)
async def test_many_jobs_release_their_threads_descriptors_processes_and_host_objects(
    method: str,
) -> None:
    """After batches of concurrent jobs, the worker holds no more threads, file descriptors,
    child processes, tasks or launcher objects than before them."""
    loop = asyncio.get_running_loop()
    # launches run on the default executor: start all its threads now, so the baseline holds them
    capacity = 8
    loop.set_default_executor(ThreadPoolExecutor(max_workers=capacity))
    barrier = threading.Barrier(capacity)
    await asyncio.gather(*(loop.run_in_executor(None, barrier.wait, 10) for _ in range(capacity)))
    launchers: list[weakref.ReferenceType[JobLauncher]] = []

    async def one_job(name: str) -> None:
        launcher = JobLauncher(
            job_entrypoint_fnc=_finishes,
            session_end_fnc=None,
            simulation_end_fnc=None,
            session_end_timeout=0.25,
            mp_ctx=mp.get_context(method),
            loop=loop,
        )
        launchers.append(weakref.ref(launcher))
        process = _core.JobSupervisor(
            launcher, JobRunnerOptions(initialize_timeout=10, close_timeout=10)
        )
        try:
            await process.start()
            await process.initialize()
            await process.launch_job(running_job(agent.Job(id=name)))
            await process.join()
            assert process.status is not None and process.status.value == "success"
            _assert_reaped(process)
        finally:
            await process.aclose()

    async def batch(index: int) -> None:
        async with asyncio.TaskGroup() as jobs:
            for slot in range(3):
                jobs.create_task(one_job(f"AJ_{index}_{slot}"))

        def launchers_released() -> bool:
            gc.collect()
            return all(reference() is None for reference in launchers)

        await until(launchers_released, "settled jobs kept their launchers alive", 5)

    parent = psutil.Process()
    await batch(0)
    # the native blocking pool retires its idle threads after ten seconds, on its own clock
    await asyncio.sleep(10)
    baseline = process_snapshot(parent)
    children = {child.pid for child in parent.children(recursive=True)}
    tasks = len(asyncio.all_tasks())

    for index in range(1, 4):
        await batch(index)
        await until(
            lambda: parent.num_threads() <= baseline["threads"], f"threads after batch {index}", 20
        )
        current = process_snapshot(parent)

        assert {child.pid for child in parent.children(recursive=True)} == children
        assert current["handles"] <= baseline["handles"]
        assert len(asyncio.all_tasks()) == tasks


# -- a real worker's job processes (`job_worker.py`) --


@pytest.mark.parametrize("executor", ["process", "thread"])
def test_each_job_runs_in_its_own_runner(job_worker: StartJobWorker, executor: str) -> None:
    """Two jobs run to their shutdown callbacks: each in a fresh process, or both on a thread of
    the worker."""
    w = job_worker("ok", jobs=2, job_executor_type=executor)

    first = json.loads(w.marker("room-0.shutdown"))
    second = json.loads(w.marker("room-1.shutdown"))

    assert w.wait() == 0, w.tail()
    if executor == "process":
        assert w.proc.pid not in (first["pid"], second["pid"])
        assert first["pid"] != second["pid"], "a job process was reused for a second job"
    else:
        assert first["pid"] == second["pid"] == w.proc.pid


@pytest.mark.parametrize("executor", ["process", "thread"])
def test_a_jobs_log_lines_and_prints_reach_the_worker_stdout(
    job_worker: StartJobWorker, executor: str
) -> None:
    """A job's log records reach stdout as JSON from its own pid (from a job process, stamped
    with its job and room), and its prints reach stdout too."""
    w = job_worker("ok", jobs=1, job_executor_type=executor)

    job = json.loads(w.marker("room-0.shutdown"))
    assert w.wait() == 0, w.tail()

    [line] = w.logged("it job says hello")
    assert (line["level"], line["name"], line["pid"]) == ("INFO", "it.job", job["pid"])
    if executor == "process":
        assert (line["job_id"], line["room"]) == (job["job_id"], "room-0")
    assert w.logged("it print from room-0")


def test_a_crashed_job_process_is_reaped_and_the_worker_takes_the_next_job(
    job_worker: StartJobWorker,
) -> None:
    """A job process dying mid-job (``os._exit``) is reaped, and the worker runs the next job."""
    w = job_worker("crash", jobs=2)

    crashed = json.loads(w.marker("room-0.started"))

    assert gone(crashed["pid"], 10)
    w.marker("room-1.shutdown")
    assert w.wait() == 0, w.tail()


def test_a_job_process_over_its_memory_limit_is_killed(
    job_worker: StartJobWorker,
) -> None:
    """A job above ``job_memory_limit_mb`` is killed, and the worker runs the next job."""
    w = job_worker("memory", jobs=2, job_memory_limit_mb=300)

    leaking = json.loads(w.marker("room-0.started"))

    assert gone(leaking["pid"], 30), "a job above job_memory_limit_mb was never killed"
    w.marker("room-1.shutdown")
    assert w.wait() == 0, w.tail()


def test_a_process_stuck_initializing_is_replaced_for_its_job(
    job_worker: StartJobWorker,
) -> None:
    """A job process still importing at ``initialize_process_timeout`` is killed, and its job
    runs on a fresh process. https://github.com/livekit/agents/pull/1556"""
    w = job_worker("slow_init", jobs=1, initialize_process_timeout=10.0, num_idle_processes=0)

    w.marker("room-0.shutdown", timeout=60)

    assert w.wait() == 0, w.tail()


def test_job_processes_exit_when_their_worker_is_killed(
    job_worker: StartJobWorker,
) -> None:
    """Job processes, busy or idle, exit once their worker is SIGKILLed."""
    w = job_worker("idle", jobs=1, num_idle_processes=2)
    job = json.loads(w.marker("room-0.started"))
    w.snapshot()
    children = [c.pid for c in psutil.Process(w.proc.pid).children(recursive=True)]
    assert job["pid"] in children

    os.kill(w.proc.pid, signal.SIGKILL)
    w.proc.wait()

    assert [pid for pid in children if not gone(pid, 20)] == []


# -- worker processes serving their text doors (`text_worker.py`) --


@pytest.fixture
def live_credentials(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    """Local startup never addresses the real storage service."""
    values = ("ws://127.0.0.1:1", "APItest", "test-secret-test-secret-test-secret")
    for name, value in zip(
        ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"), values, strict=True
    ):
        monkeypatch.setenv(name, value)
    return values


async def test_worker_processes_each_serve_their_text_door(workers: Workers) -> None:
    """Two worker processes each answer on their own listener, their text door included (an
    empty JSON-RPC body gets the protocol's invalid-request error)."""
    for name, caller in (("worker-a", workers.a), ("worker-b", workers.b)):
        async with httpx.AsyncClient(base_url=caller.http.base_url, trust_env=False) as http:
            ready = await http.get("/ready")
            door = await http.post("/desk/v1", json={})

        assert ready.text == name
        assert door.status_code == 200
        assert door.json()["error"]["code"] == -32600


def test_a_worker_shutdown_leaves_no_coroutine_unawaited(job_worker: StartJobWorker) -> None:
    """A worker that runs its jobs and exits warns of no host coroutine left unawaited."""
    w = job_worker("ok", jobs=2)

    w.marker("room-1.shutdown")

    assert w.wait() == 0, w.tail()
    assert not w.logged("was never awaited"), w.tail()
