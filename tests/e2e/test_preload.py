"""Application preloads (``AgentServer(preload_modules=...)``) run in each job process before it is
ready, and on Linux in the forkserver jobs fork from. The job runners are built as the worker
builds them (``processes.job_runner``). Run as a script, this file is the forkserver probe."""

from __future__ import annotations

import asyncio
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import psutil
import pytest

from livekit.agents import _core
from livekit.agents.ipc import configure_forkserver
from livekit.agents.job import JobContext
from livekit.protocol import agent
from tests.e2e.processes import (
    OwnRunner,
    close_owned_executor,
    job_runner,
    run_owned_process,
    running_job,
)
from tests.fakes import until

from .resources import StartJobWorker


def _record(*observation: object) -> None:
    with Path(os.environ["PRELOAD_EVENTS_PATH"]).open("a") as events:
        events.write(json.dumps(list(observation)) + "\n")


async def _observe(context: JobContext) -> None:
    """The first job in a process: what was ready before its loop, then a private mutation."""
    from livekit.rtc._ffi_client import FfiClient

    first = importlib.import_module("_livekit_preload_first")
    initial = first.table[0]
    first.table[0] = 99
    _record(
        "job",
        os.getpid(),
        multiprocessing.get_start_method(),
        FfiClient._instance is not None,
        initial,
    )


async def _cow(context: JobContext) -> None:
    """Writing every page of a forkserver-preloaded table copies it into this process only."""
    first = importlib.import_module("_livekit_preload_first")
    initial = first.table[0]
    memory_before = psutil.Process().memory_full_info().uss
    for offset in range(0, len(first.table), os.sysconf("SC_PAGE_SIZE")):
        first.table[offset] = 99
    first.table[-1] = 99
    memory_after = psutil.Process().memory_full_info().uss
    _record("cow", os.getpid(), initial, len(first.table), memory_before, memory_after)


@pytest.fixture
def preload_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("PRELOAD_EVENTS_PATH", str(events))
    # This suite observes process bootstrapping; model preparation has its own warmup gate.
    monkeypatch.setenv("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "_livekit_preload_first.py").write_text(
        "import os, json\n"
        "from pathlib import Path\n"
        "table = bytearray(b'\\x07' * (16 * 1024 * 1024))\n"
        "with Path(os.environ['PRELOAD_EVENTS_PATH']).open('a') as output:\n"
        "    output.write(json.dumps(['first', os.getpid()]) + '\\n')\n"
    )
    (tmp_path / "_livekit_preload_second.py").write_text(
        "import os, json\n"
        "from pathlib import Path\n"
        "with Path(os.environ['PRELOAD_EVENTS_PATH']).open('a') as output:\n"
        "    output.write(json.dumps(['second', os.getpid()]) + '\\n')\n"
    )
    (tmp_path / "_livekit_preload_broken.py").write_text(
        "raise ImportError('application weights unavailable')\n"
    )
    (tmp_path / "_livekit_preload_threads.py").write_text(
        "import threading\n"
        "threading.Thread(target=threading.Event().wait, name='preload_background', daemon=True).start()\n"
    )
    return events


def _executor(
    own_executor: OwnRunner,
    *,
    modules: tuple[str, ...],
    entrypoint: Callable[[JobContext], Awaitable[None]] = _observe,
    method: str = "spawn",
) -> _core.JobSupervisor:
    return own_executor(
        job_runner(
            entrypoint,
            mp_ctx=multiprocessing.get_context(method),
            preload_modules=modules,
            initialize_timeout=15,
            close_timeout=1,
            ping_interval=0.1,
            ping_timeout=5,
            high_ping_threshold=1,
        )
    )


async def _run_job(process: _core.JobSupervisor, events: Path, kind: str) -> list[object]:
    """Launch a fake job and return the observation its entrypoint recorded."""
    await process.launch_job(running_job(agent.Job(id=kind)))

    def recorded() -> list[list[object]]:
        lines = events.read_text().splitlines() if events.exists() else []
        return [o for o in map(json.loads, lines) if o[0] == kind and o[1] == process.pid]

    await until(recorded, f"the {kind} job recorded nothing", 20)
    return recorded()[0]


def _kinds(events: Path) -> list[object]:
    return [json.loads(line)[0] for line in events.read_text().splitlines()]


async def _close(process: _core.JobSupervisor) -> None:
    await process.aclose()
    await process.join()


async def test_preloads_rtc_and_the_start_method_are_ready_before_the_first_job(
    own_executor: OwnRunner, preload_events: Path
) -> None:
    """The first job finds its preloads imported, the RTC runtime started and the server's
    default start method in use."""
    # the server's default start method (`multiprocessing_context`)
    default = "forkserver" if sys.platform == "linux" else "spawn"
    process = _executor(own_executor, modules=("_livekit_preload_first",), method=default)
    await process.start()
    await process.initialize()

    _, pid, method, rtc_warm, _ = await _run_job(process, preload_events, "job")

    assert (pid, method) == (process.pid, default)
    assert rtc_warm, "the RTC runtime would initialize inside the first job"
    await _close(process)


async def test_spawn_preloads_in_order_before_ready_with_independent_data(
    own_executor: OwnRunner, preload_events: Path
) -> None:
    """Each spawned process imports its preloads in order before it is ready, and one job's
    change to preloaded data never reaches another process."""
    pids = []
    for _ in range(2):
        process = _executor(
            own_executor, modules=("_livekit_preload_first", "_livekit_preload_second")
        )
        await process.start()
        await process.initialize()
        _record("ready", process.pid)
        initial = (await _run_job(process, preload_events, "job"))[4]
        assert initial == 7, "another process's mutation reached this one"
        pids.append(process.pid)
        await _close(process)

    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert _kinds(preload_events) == ["first", "second", "ready", "job"] * 2
    assert pids[0] != pids[1]
    for batch, pid in zip((observations[:4], observations[4:]), pids, strict=True):
        assert all(observation[1] == pid for observation in batch)


async def test_a_failed_preload_names_its_module_and_its_process_is_reaped(
    own_executor: OwnRunner, preload_events: Path
) -> None:
    """A preload raising fails the process's initialization with the module and its error,
    stops the later preloads, and the process is reaped."""
    process = _executor(own_executor, modules=("_livekit_preload_first", "_livekit_preload_broken"))
    await process.start()

    with pytest.raises(Exception, match="_livekit_preload_broken.*application weights unavailable"):
        await process.initialize()
    await process.join()

    assert process.exitcode is not None and process.exitcode != 0
    assert process.pid is not None and not psutil.pid_exists(process.pid)
    assert _kinds(preload_events) == ["first"]


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
async def test_forkserver_preloads_share_pages_and_keep_child_mutations_private(
    preload_events: Path,
) -> None:
    """Forked jobs share the forkserver's preloaded pages until they write them, and a job's
    write stays its own."""
    result = await _run_forkserver_probe(preload_events, "normal")
    assert result.returncode == 0, result.stdout
    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert _kinds(preload_events) == ["first", "second", "ready", "cow", "ready", "cow"]
    preload_pid = observations[0][1]
    assert observations[1][1] == preload_pid
    child_pids = [observations[2][1], observations[4][1]]
    assert child_pids[0] != child_pids[1] and preload_pid not in child_pids
    for ready, cow in (observations[2:4], observations[4:6]):
        assert cow[1] == ready[1]
        initial, payload, before, after = cow[2:]
        assert initial == 7, "a sibling modified the forkserver's table"
        assert after - before >= payload // 2, "the table was already private before the job"


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
@pytest.mark.parametrize("failure", ["import", "threads"])
async def test_unsafe_forkserver_preload_refuses_job_forks(
    preload_events: Path, failure: str
) -> None:
    """A forkserver preload that fails to import or leaves a thread running refuses to fork
    jobs, naming the module or the thread."""
    result = await _run_forkserver_probe(preload_events, failure)
    assert result.returncode != 0, result.stdout
    if failure == "import":
        assert "_livekit_preload_broken" in result.stdout
        assert "application weights unavailable" in result.stdout
    else:
        assert "preload_background" in result.stdout and "refusing to fork" in result.stdout
    assert _kinds(preload_events) == ["first"]


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
def test_forkserver_job_processes_run_from_a_single_threaded_forkserver(
    job_worker: StartJobWorker,
) -> None:
    """A job process is forked from a single-threaded forkserver: a fork of a multi-threaded
    process can deadlock the child."""
    w = job_worker("forkserver", jobs=1, multiprocessing_context="forkserver")
    result = json.loads(w.marker("room-0.shutdown"))
    assert result["parent"] != w.proc.pid, "the job process was not forked by a forkserver"
    assert result["parent_threads"] == 1, result


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
def test_forkserver_freezes_its_preloads_before_forking_jobs() -> None:
    """The forkserver freezes its preloaded objects (no copy-on-write from the collector) while a
    forked job still collects its own cycles."""
    result = subprocess.run(
        [sys.executable, "-m", "tests.e2e.forkserver_freeze"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["freeze_count"] > 0
    assert outcome["child_cycle_collected"]


async def _run_forkserver_probe(
    preload_events: Path, mode: str
) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "FORKSERVER_PRELOAD_PROBE": mode,
        "PYTHONPATH": os.pathsep.join(
            [
                str(preload_events.parent),
                str(repository / "python" / "livekit-agents"),
                str(repository),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    }
    return await asyncio.to_thread(
        run_owned_process,
        [sys.executable, __file__, str(preload_events.parent / "forkserver-driver")],
        preload_events.parent / "forkserver-watchdog",
        env,
        45,
    )


async def _forkserver_probe(artifacts: Path) -> None:
    context = multiprocessing.get_context("forkserver")
    mode = os.environ["FORKSERVER_PRELOAD_PROBE"]
    modules = (
        "_livekit_preload_first",
        {
            "normal": "_livekit_preload_second",
            "import": "_livekit_preload_broken",
            "threads": "_livekit_preload_threads",
        }[mode],
    )
    configure_forkserver(context, modules)
    events = Path(os.environ["PRELOAD_EVENTS_PATH"])
    for _ in range(2):
        process = _executor(
            lambda executor: executor, modules=modules, entrypoint=_cow, method="forkserver"
        )
        try:
            await process.start()
            await process.initialize()
            _record("ready", process.pid)
            await _run_job(process, events, "cow")
        finally:
            await close_owned_executor(process, artifacts)


if __name__ == "__main__":
    asyncio.run(_forkserver_probe(Path(sys.argv[1])))
