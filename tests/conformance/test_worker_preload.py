"""Application preloads and per-process setup through the real multiprocessing adapter."""

from __future__ import annotations

import asyncio
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import psutil
import pytest

from livekit.agents import AgentServer
from livekit.agents.ipc import JobExecutor
from livekit.agents.ipc.launch import configure_forkserver
from livekit.agents.job import JobContext, JobProcess
from tests.conformance.ipc_processes import close_owned_executor, run_owned_process

pytestmark = [pytest.mark.e2e, pytest.mark.no_concurrent]


async def _entrypoint(context: JobContext) -> None:
    pass


def _setup(process: JobProcess) -> None:
    first = importlib.import_module("_livekit_preload_first")
    initial = first.table[0]
    first.table[0] = 99
    with Path(os.environ["PRELOAD_EVENTS_PATH"]).open("a") as events:
        events.write(json.dumps(["setup", os.getpid(), initial, process.pid]) + "\n")


def _failed_setup(process: JobProcess) -> None:
    _setup(process)
    raise ValueError("per-process setup refused")


def _record_start_method(events_path: Path, process: JobProcess) -> None:
    with events_path.open("a") as events:
        events.write(json.dumps(["method", process.pid, multiprocessing.get_start_method()]) + "\n")


def _awaitable_setup(process: JobProcess) -> object:
    _setup(process)

    async def setup() -> None:
        raise AssertionError("an asynchronous setup body must not execute")

    return setup()


def _cow_setup(process: JobProcess) -> None:
    first = importlib.import_module("_livekit_preload_first")
    memory_before = psutil.Process().memory_full_info().uss
    _setup(process)
    for offset in range(0, len(first.table), os.sysconf("SC_PAGE_SIZE")):
        first.table[offset] = 99
    first.table[-1] = 99
    memory_after = psutil.Process().memory_full_info().uss
    with Path(os.environ["PRELOAD_EVENTS_PATH"]).open("a") as events:
        events.write(
            json.dumps(["cow", os.getpid(), len(first.table), memory_before, memory_after]) + "\n"
        )


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
    setup: Callable[[JobProcess], None],
    method: str | None = "spawn",
) -> JobExecutor:
    return own_executor(
        JobExecutor(
            job_entrypoint_fnc=_entrypoint,
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
            setup_fnc=setup,
        )
    )


@pytest.mark.asyncio
async def test_start_method_defaults_launch_the_expected_child(
    own_executor: Callable[[JobExecutor], JobExecutor], preload_events: Path
) -> None:
    process = _executor(
        own_executor,
        modules=(),
        setup=partial(_record_start_method, preload_events),
        method=None,
    )
    await asyncio.wait_for(process.start(), timeout=20)
    await asyncio.wait_for(process.initialize(), timeout=20)
    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert observations == [
        ["method", process.pid, "forkserver" if sys.platform == "linux" else "spawn"]
    ]
    await asyncio.wait_for(process.aclose(), timeout=5)
    await asyncio.wait_for(process.join(), timeout=5)
    assert process.exitcode == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["server", "executor"])
async def test_async_setup_is_refused_at_the_public_launch_boundary(
    own_executor: Callable[[JobExecutor], JobExecutor], boundary: str
) -> None:
    async def setup(process: JobProcess) -> None:
        raise AssertionError("asynchronous setup must not execute")

    callback = cast(Callable[[JobProcess], None], setup)
    with pytest.raises(TypeError, match="setup_fnc must be a synchronous callable"):
        if boundary == "server":
            AgentServer(setup_fnc=callback)
        else:
            _executor(own_executor, modules=(), setup=callback)


@pytest.mark.asyncio
async def test_spawn_preloads_before_setup_and_ready_with_independent_data(
    own_executor: Callable[[JobExecutor], JobExecutor], preload_events: Path
) -> None:
    pids = []
    for _ in range(2):
        process = _executor(
            own_executor,
            modules=("_livekit_preload_first", "_livekit_preload_second"),
            setup=_setup,
        )
        await asyncio.wait_for(process.start(), timeout=20)
        await asyncio.wait_for(process.initialize(), timeout=20)
        pids.append(process.pid)
        with preload_events.open("a") as events:
            events.write(json.dumps(["ready", process.pid]) + "\n")
        await asyncio.wait_for(process.aclose(), timeout=5)
        await asyncio.wait_for(process.join(), timeout=5)
        assert process.exitcode == 0

    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert [observation[0] for observation in observations] == [
        "first",
        "second",
        "setup",
        "ready",
        "first",
        "second",
        "setup",
        "ready",
    ]
    assert pids[0] != pids[1]
    for batch, pid in zip((observations[:4], observations[4:]), pids, strict=True):
        assert all(observation[1] == pid for observation in batch)
        assert batch[2][2:] == [7, pid], "another process's mutation reached this setup"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["preload", "setup", "awaitable"])
async def test_process_initialization_failure_names_cause_and_prevents_ready(
    own_executor: Callable[[JobExecutor], JobExecutor],
    preload_events: Path,
    failure: str,
    capfd: pytest.CaptureFixture[str],
) -> None:
    setup = {
        "preload": _setup,
        "setup": _failed_setup,
        "awaitable": cast(Callable[[JobProcess], None], _awaitable_setup),
    }[failure]
    process = _executor(
        own_executor,
        modules=("_livekit_preload_first", "_livekit_preload_broken")
        if failure == "preload"
        else ("_livekit_preload_first",),
        setup=setup,
    )
    await asyncio.wait_for(process.start(), timeout=20)
    expected = {
        "preload": "_livekit_preload_broken.*application weights unavailable",
        "setup": "per-process setup refused",
        "awaitable": "setup_fnc returned an awaitable",
    }[failure]
    with pytest.raises(Exception, match=expected):
        await asyncio.wait_for(process.initialize(), timeout=20)
    await asyncio.wait_for(process.join(), timeout=5)
    assert process.exitcode is not None and process.exitcode != 0
    observations = [json.loads(line) for line in preload_events.read_text().splitlines()]
    assert [observation[0] for observation in observations] == (
        ["first"] if failure == "preload" else ["first", "setup"]
    )
    if failure == "awaitable":
        captured = capfd.readouterr()
        assert "was never awaited" not in captured.err


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
        "setup",
        "cow",
        "ready",
        "setup",
        "cow",
        "ready",
    ]
    preload_pid = observations[0][1]
    assert observations[1][1] == preload_pid
    child_pids = [observations[2][1], observations[5][1]]
    assert child_pids[0] != child_pids[1] and preload_pid not in child_pids
    for setup, cow, ready in (observations[2:5], observations[5:8]):
        assert setup[2:] == [7, setup[1]], "a sibling modified the forkserver's table"
        assert cow[1] == setup[1] == ready[1]
        payload, before, after = cow[2:]
        assert after - before >= payload // 2, "the table was already private before setup"


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
    for _ in range(2):
        process = _executor(
            lambda executor: executor, modules=modules, setup=_cow_setup, method="forkserver"
        )
        try:
            await asyncio.wait_for(process.start(), timeout=20)
            await asyncio.wait_for(process.initialize(), timeout=20)
            with Path(os.environ["PRELOAD_EVENTS_PATH"]).open("a") as events:
                events.write(json.dumps(["ready", process.pid]) + "\n")
        finally:
            await close_owned_executor(process, artifacts)


if __name__ == "__main__":
    asyncio.run(_forkserver_probe(Path(sys.argv[1])))
