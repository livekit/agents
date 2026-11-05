"""Application preloads through the real multiprocessing adapter, observed from a job."""

from __future__ import annotations

import asyncio
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import psutil
import pytest

from livekit.agents.ipc import JobExecutor, configure_forkserver
from livekit.agents.job import JobAcceptArguments, JobContext, RunningJobInfo
from livekit.protocol import agent
from tests.conformance.ipc_processes import close_owned_executor, run_owned_process

pytestmark = [pytest.mark.e2e, pytest.mark.no_concurrent]


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
    own_executor: Callable[[JobExecutor], JobExecutor],
    *,
    modules: tuple[str, ...],
    entrypoint: Callable[[JobContext], Awaitable[None]] = _observe,
    method: str | None = "spawn",
) -> JobExecutor:
    return own_executor(
        JobExecutor(
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=None,
            simulation_end_fnc=None,
            initialize_timeout=15,
            close_timeout=1,
            session_end_timeout=1,
            ping_interval=0.1,
            ping_timeout=5,
            high_ping_threshold=1,
            http_proxy=None,
            mp_ctx=multiprocessing.get_context(method) if method is not None else None,
            loop=asyncio.get_running_loop(),
            preload_modules=modules,
        )
    )


async def _run_job(process: JobExecutor, events: Path, kind: str) -> list[object]:
    """Launch a fake job and return the observation its entrypoint recorded."""
    await process.launch_job(
        RunningJobInfo(JobAcceptArguments("", "", ""), agent.Job(id=kind), "", "", "", True)
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        for line in events.read_text().splitlines() if events.exists() else []:
            observation: list[object] = json.loads(line)
            if observation[0] == kind and observation[1] == process.pid:
                return observation
        await asyncio.sleep(0.05)
    raise AssertionError(f"the {kind} job recorded nothing")


async def _close(process: JobExecutor) -> None:
    await asyncio.wait_for(process.aclose(), timeout=5)
    await asyncio.wait_for(process.join(), timeout=5)


@pytest.mark.asyncio
async def test_preloads_rtc_and_the_start_method_are_ready_before_the_first_job(
    own_executor: Callable[[JobExecutor], JobExecutor], preload_events: Path
) -> None:
    process = _executor(own_executor, modules=("_livekit_preload_first",), method=None)
    await asyncio.wait_for(process.start(), timeout=20)
    await asyncio.wait_for(process.initialize(), timeout=20)
    _, pid, method, rtc_warm, _ = await _run_job(process, preload_events, "job")
    assert pid == process.pid
    assert method == ("forkserver" if sys.platform == "linux" else "spawn")
    assert rtc_warm, "the RTC runtime would initialize inside the first job"
    await _close(process)


@pytest.mark.asyncio
async def test_spawn_preloads_in_order_before_ready_with_independent_data(
    own_executor: Callable[[JobExecutor], JobExecutor], preload_events: Path
) -> None:
    pids = []
    for _ in range(2):
        process = _executor(
            own_executor, modules=("_livekit_preload_first", "_livekit_preload_second")
        )
        await asyncio.wait_for(process.start(), timeout=20)
        await asyncio.wait_for(process.initialize(), timeout=20)
        _record("ready", process.pid)
        initial = (await _run_job(process, preload_events, "job"))[4]
        assert initial == 7, "another process's mutation reached this one"
        pids.append(process.pid)
        await _close(process)

    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert [observation[0] for observation in observations] == [
        "first",
        "second",
        "ready",
        "job",
        "first",
        "second",
        "ready",
        "job",
    ]
    assert pids[0] != pids[1]
    for batch, pid in zip((observations[:4], observations[4:]), pids, strict=True):
        assert all(observation[1] == pid for observation in batch)


@pytest.mark.asyncio
async def test_a_failed_preload_names_its_module_and_prevents_ready(
    own_executor: Callable[[JobExecutor], JobExecutor], preload_events: Path
) -> None:
    process = _executor(own_executor, modules=("_livekit_preload_first", "_livekit_preload_broken"))
    await asyncio.wait_for(process.start(), timeout=20)
    with pytest.raises(Exception, match="_livekit_preload_broken.*application weights unavailable"):
        await asyncio.wait_for(process.initialize(), timeout=20)
    await asyncio.wait_for(process.join(), timeout=5)
    assert process.exitcode is not None and process.exitcode != 0
    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert [observation[0] for observation in observations] == ["first"]


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
@pytest.mark.asyncio
async def test_forkserver_preloads_share_pages_and_keep_child_mutations_private(
    preload_events: Path,
) -> None:
    result = await _run_forkserver_probe(preload_events, "normal")
    assert result.returncode == 0, result.stdout
    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert [observation[0] for observation in observations] == [
        "first",
        "second",
        "ready",
        "cow",
        "ready",
        "cow",
    ]
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
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["import", "threads"])
async def test_unsafe_forkserver_preload_refuses_job_forks(
    preload_events: Path, failure: str
) -> None:
    result = await _run_forkserver_probe(preload_events, failure)
    assert result.returncode != 0, result.stdout
    if failure == "import":
        assert "_livekit_preload_broken" in result.stdout
        assert "application weights unavailable" in result.stdout
    else:
        assert "preload_background" in result.stdout and "refusing to fork" in result.stdout
    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert [observation[0] for observation in observations] == ["first"]


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
                str(repository / "livekit-agents-next"),
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
            await asyncio.wait_for(process.start(), timeout=20)
            await asyncio.wait_for(process.initialize(), timeout=20)
            _record("ready", process.pid)
            await _run_job(process, events, "cow")
        finally:
            await close_owned_executor(process, artifacts)


if __name__ == "__main__":
    asyncio.run(_forkserver_probe(Path(sys.argv[1])))
