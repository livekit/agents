"""The job processes a real ``AgentServer`` runs: spawned, supervised, killed and drained by the
core, on the machine's own processes and sockets, which the voice DST does not model.

Each test runs this file as a worker (``python tests/test_it_ipc.py``, unregistered, jobs
simulated with ``fake_job``) and reads what it and its jobs left: marker files written by the
entrypoint and the JSON log lines on the worker's stdout. The worker is the same file so the job
processes import the entrypoint from it, as they import a user's agent file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from multiprocessing import current_process
from pathlib import Path
from typing import Any

import pytest

SCENARIO = os.environ.get("IT_SCENARIO", "")
MARKERS = Path(os.environ.get("IT_DIR", "."))

# a job process imports this file before it answers the initialize request: the first one to do
# so in the `slow_init` scenario takes longer than the server's initialize_process_timeout
if SCENARIO == "slow_init" and current_process().name == "job_proc":
    try:
        (MARKERS / "slow_init").open("x").close()
        time.sleep(60)
    except FileExistsError:
        pass


# ============================================================================ the tests

pytestmark = pytest.mark.e2e

_core = pytest.importorskip("livekit.agents._core", reason="needs livekit-agents-next")
psutil = pytest.importorskip("psutil")

if sys.platform == "win32":
    pytest.skip("POSIX process management", allow_module_level=True)


class Worker:
    """This file run as a worker in a subprocess, its stdout collected as JSON log lines."""

    def __init__(self, tmp_path: Path, scenario: str, jobs: int, **options: Any) -> None:
        self.dir = tmp_path
        env = {
            **os.environ,
            "IT_SCENARIO": scenario,
            "IT_DIR": str(tmp_path),
            "IT_JOBS": str(jobs),
            "IT_OPTS": json.dumps(options),
            "PYTHONUNBUFFERED": "1",
        }
        self.out = (tmp_path / "stdout").open("w+")
        self.proc = subprocess.Popen(
            [sys.executable, __file__],
            env=env,
            stdout=self.out,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parents[1]),
            start_new_session=True,
        )
        self.owned: dict[int, float] = {}

    def lines(self) -> list[dict[str, Any]]:
        lines = []
        for line in (self.dir / "stdout").read_text().splitlines():
            try:
                lines.append(json.loads(line))
            except ValueError:
                lines.append({"raw": line})
        return lines

    def logged(self, message: str) -> list[dict[str, Any]]:
        return [rec for rec in self.lines() if message in rec.get("message", rec.get("raw", ""))]

    def marker(self, name: str, timeout: float = 30.0) -> str:
        path = self.dir / name
        deadline = time.monotonic() + timeout
        while not path.exists():
            self.snapshot()
            if self.proc.poll() is not None:
                pytest.fail(f"worker exited ({self.proc.returncode}) before {name}:\n{self.tail()}")
            if time.monotonic() > deadline:
                pytest.fail(f"no {name} after {timeout}s:\n{self.tail()}")
            time.sleep(0.05)
        return path.read_text()

    def wait(self, timeout: float = 60.0) -> int:
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"worker still running after {timeout}s:\n{self.tail()}")

    def tail(self) -> str:
        return "\n".join(json.dumps(x) for x in self.lines()[-40:])

    def snapshot(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            root = psutil.Process(self.proc.pid)
            children = [root, *root.children(recursive=True)]
            for process in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    self.owned[process.pid] = process.create_time()
                    records.append(
                        {
                            "pid": process.pid,
                            "created": process.create_time(),
                            "status": process.status(),
                            "threads": process.num_threads(),
                            "fds": process.num_fds(),
                        }
                    )
        except psutil.NoSuchProcess:
            pass
        return records

    def close(self) -> None:
        before = self.snapshot()
        artifact = self.dir / "process-cleanup.json"
        artifact.write_text(json.dumps({"before": before, "known": self.owned}, indent=2))
        # The session includes forkserver/tracker and descendants even after logical-parent death.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=5)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            retained = []
            for pid, created in self.owned.items():
                with contextlib.suppress(psutil.NoSuchProcess):
                    process = psutil.Process(pid)
                    if process.create_time() == created:
                        retained.append({"pid": pid, "status": process.status()})
            if not retained:
                break
            time.sleep(0.02)
        artifact.write_text(
            json.dumps({"before": before, "known": self.owned, "after": retained}, indent=2)
        )
        self.out.close()
        assert not retained, f"owned processes remain (zombies are not reaped): {artifact}"


@pytest.fixture
def worker(tmp_path: Path):
    started: list[Worker] = []

    def start(scenario: str, jobs: int = 1, **options: Any) -> Worker:
        w = Worker(tmp_path, scenario, jobs, **options)
        started.append(w)
        return w

    yield start
    for w in started:
        w.close()


def _gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # A zombie has exited but still consumes a process-table entry until reaped.
            psutil.Process(pid).status()
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("executor", ["process", "thread"])
def test_job_runs_and_its_logs_reach_stdout(worker, executor: str) -> None:
    """Guards a job whose entrypoint never runs, whose shutdown callbacks are skipped, whose log
    lines or prints are lost (or, from a job process, miss their job fields), or after which the
    worker takes no second job."""
    w = worker("ok", jobs=2, job_executor_type=executor)
    first = json.loads(w.marker("room-0.shutdown"))
    second = json.loads(w.marker("room-1.shutdown"))
    assert w.wait() == 0, w.tail()

    worker_pid = w.proc.pid
    if executor == "process":
        assert first["pid"] != worker_pid and second["pid"] != worker_pid
        assert first["pid"] != second["pid"], "a job process is reused for a second job"
    else:
        assert first["pid"] == second["pid"] == worker_pid

    for room, job in (("room-0", first), ("room-1", second)):
        (line,) = [x for x in w.logged("it job says hello") if x.get("it_field") == room]
        assert line["level"] == "INFO" and line["name"] == "it.job", line
        assert line["pid"] == job["pid"], line
        if executor == "process":
            # a job thread's lines carry no job fields (the reference's neither)
            assert (line["job_id"], line["room"]) == (job["job_id"], room), line
        assert w.logged(f"it print from {room}"), "a job's print() never reached stdout"


def test_crashed_job_process_is_reaped_and_worker_keeps_serving(worker) -> None:
    """Guards a job process dying mid-job (a native crash, os._exit) wedging the pool or taking
    the worker down: the worker must reap it and run the next job."""
    w = worker("crash", jobs=2)
    crashed = json.loads(w.marker("room-0.started"))
    assert _gone(crashed["pid"], 10)
    w.marker("room-1.shutdown")
    assert w.wait() == 0, w.tail()
    assert not (w.dir / "room-0.shutdown").exists()


def test_hung_job_process_is_killed_after_ping_timeout(worker) -> None:
    """Guards a job whose event loop is blocked (sync I/O, a busy loop) living forever: the core
    must stop getting pongs, kill it, and keep serving."""
    w = worker("hang", jobs=2, it_ping_timeout=3.0)
    hung = json.loads(w.marker("room-0.started"))
    assert _gone(hung["pid"], 20), "a job blocking its event loop was never killed"
    assert w.logged("process is unresponsive, stopping it"), w.tail()
    w.marker("room-1.shutdown")
    assert w.wait() == 0, w.tail()


def test_job_process_over_memory_limit_is_killed(worker) -> None:
    """Guards job_memory_limit_mb not being enforced: a leaking job must be killed (and said so)
    before it takes the host's memory."""
    w = worker("memory", jobs=2, job_memory_limit_mb=300)
    leaking = json.loads(w.marker("room-0.started"))
    assert _gone(leaking["pid"], 30), "a job above job_memory_limit_mb was never killed"
    (line,) = w.logged("process exceeded memory limit, stopping it")
    assert line["memory_usage_mb"] > 300, line
    w.marker("room-1.shutdown")
    assert w.wait() == 0, w.tail()


def test_slow_initialization_is_killed_and_job_runs_on_next_process(worker) -> None:
    """Guards a job process stuck initializing (a heavy import at the agent file's top level)
    hanging the job that waits for it: the core must kill it at initialize_process_timeout and
    hand the job to a fresh process."""
    w = worker("slow_init", jobs=1, initialize_process_timeout=10.0, num_idle_processes=0)
    w.marker("room-0.shutdown", timeout=60)
    assert w.wait() == 0, w.tail()
    (line,) = w.logged("error initializing process")
    assert "initialization timed out" in line["exc_info"], line


def test_job_processes_exit_when_the_worker_is_killed(worker) -> None:
    """Guards orphaned job processes (and the rooms they hold) outliving a worker that was
    SIGKILLed or OOM-killed: a child must notice its parent's socket close and exit."""
    w = worker("idle", jobs=1, num_idle_processes=2)
    job = json.loads(w.marker("room-0.started"))
    w.snapshot()
    children = [c.pid for c in psutil.Process(w.proc.pid).children(recursive=True)]
    assert job["pid"] in children
    os.kill(w.proc.pid, signal.SIGKILL)
    w.proc.wait()
    lingering = [pid for pid in children if not _gone(pid, 20)]
    assert not lingering, f"processes outlived their worker: {lingering}"


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
def test_forkserver_job_processes_run_from_a_single_threaded_forkserver(worker) -> None:
    """Guards job processes forked from the forkserver (Linux's default) failing on state they
    inherit from it, and a preload starting a thread there: a fork of a multi-threaded process
    can deadlock the child."""
    w = worker("forkserver", jobs=1, multiprocessing_context="forkserver")
    result = json.loads(w.marker("room-0.shutdown"))
    assert result["parent"] != w.proc.pid, "the job process was not forked by a forkserver"
    assert result["parent_threads"] == 1, result


# ============================================================================ the worker


def _write(name: str, data: dict[str, Any]) -> None:
    tmp = MARKERS / f".{name}"
    tmp.write_text(json.dumps(data))
    tmp.rename(MARKERS / name)


async def _entrypoint(ctx: Any) -> None:
    import logging

    room = ctx.job.room.name
    info = {"pid": os.getpid(), "job_id": ctx.job.id}

    async def on_shutdown(reason: str) -> None:
        _write(f"{room}.shutdown", {**info, "reason": reason, **extra})

    ctx.add_shutdown_callback(on_shutdown)
    extra: dict[str, Any] = {}
    _write(f"{room}.started", info)

    scenario = SCENARIO if room == "room-0" else "ok"
    if scenario == "crash":
        os._exit(7)
    elif scenario == "hang":
        time.sleep(600)
    elif scenario == "idle":
        await asyncio.sleep(600)
    elif scenario == "memory":
        hog = bytearray(800 * 1024 * 1024)
        for i in range(0, len(hog), 4096):
            hog[i] = 1
        await asyncio.sleep(600)
    elif scenario == "forkserver":
        parent = psutil.Process(os.getppid())
        extra.update(parent=parent.pid, parent_threads=parent.num_threads())

    logging.getLogger("it.job").info("it job says hello", extra={"it_field": room})
    print(f"it print from {room}", flush=True)
    ctx.shutdown("it done")


async def _drive(server: Any, jobs: int) -> None:
    started = asyncio.Event()
    server._on_started.append(started.set)
    run = asyncio.create_task(server._run(devmode=False, unregistered=True))
    readiness = asyncio.create_task(started.wait())
    try:
        await asyncio.wait({run, readiness}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
        if run.done():
            run.result()
        if not started.is_set():
            raise TimeoutError("the worker did not become ready")
        for i in range(jobs):
            room = f"room-{i}"
            await server._simulate_job(room, fake_job=True)
            # Start the next job only after the previous one's actual lifecycle settled.
            while True:
                began = (MARKERS / f"{room}.started").exists() or (
                    MARKERS / f"{room}.shutdown"
                ).exists()
                if began and not server._active_jobs:
                    break
                await asyncio.sleep(0.05)
    finally:
        readiness.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await readiness
        await server.aclose()
        await run


def _worker_main() -> None:
    from livekit.agents import AgentServer, cli, ipc
    from livekit.agents.job import JobExecutorType

    options = json.loads(os.environ["IT_OPTS"])
    ping_timeout = options.pop("it_ping_timeout", None)
    if ping_timeout is not None:
        init = ipc.ProcPool.__init__

        def patched(self: Any, *args: Any, **kwargs: Any) -> None:
            init(self, *args, **kwargs)
            self._executor_args["ping_timeout"] = ping_timeout

        ipc.ProcPool.__init__ = patched  # type: ignore[method-assign]
    if "job_executor_type" in options:
        options["job_executor_type"] = JobExecutorType(options["job_executor_type"])
    options.setdefault("num_idle_processes", 1)
    options.setdefault("multiprocessing_context", "spawn")
    options.setdefault("port", 0)

    # the IPC options are private: set them where the CLI does
    server = AgentServer()
    server._update_options(**options)
    server.rtc_session()(_entrypoint)
    cli.setup_logging("DEBUG", "json")
    asyncio.run(_drive(server, int(os.environ["IT_JOBS"])))


if __name__ == "__main__":
    _worker_main()
