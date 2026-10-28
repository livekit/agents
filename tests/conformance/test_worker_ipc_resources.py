"""Real multiprocessing launch, native health supervision and independent reaping receipts."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
import weakref
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil
import pytest

from livekit.agents import _core
from livekit.agents._generated.worker_ipc import (
    IpcFailureIo,
    IpcFailureMalformed,
    IpcFailureTooLarge,
    ProcessCompletion,
)
from livekit.agents.ipc import JobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent

pytestmark = pytest.mark.e2e

_METHODS = ["spawn", *(["forkserver"] if sys.platform == "linux" else [])]
_CONCURRENT_JOBS = 3
_PROCESS_INITIALIZE_TIMEOUT = 10.0
_PROCESS_CLOSE_TIMEOUT = 10.0
# Tokio's blocking pool retires idle workers after ten seconds when no override is set.
_NATIVE_CACHE_IDLE_TIMEOUT = 10.0


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        psutil.Process(pid).kill()
    else:
        os.kill(pid, signal.SIGKILL)


class _ChildHost:
    async def _core_start_job(self, fields: _core.StartJobFields) -> None:
        # The marker proves the host callback entered before its deliberate event-loop stall.
        Path(fields.metadata).write_text("entered")
        time.sleep(60)

    async def _core_dump_stacks(self) -> None:
        pass


def _ipc_child(endpoint: socket.socket, behavior: str) -> None:
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: None)
    if behavior.startswith("wire_"):
        endpoint.recv(4)  # wait until the parent admitted Initialize
        frames = {
            "wire_malformed": bytes([0, 0, 0, 1, 99]),
            "wire_truncated": bytes([0, 0, 0, 5, 1]),
            "wire_oversized_prefix": (2**32 - 1).to_bytes(4, "big"),
            "wire_eof": b"",
        }
        endpoint.sendall(frames[behavior])
        if behavior == "wire_oversized_prefix":
            time.sleep(60)  # no payload: refusal must follow the prefix alone
        endpoint.close()
        os._exit(0)
    _core.init(_core.Role.Job)

    def initialize(proxy: str | None) -> None:
        if behavior == "failed_initialize":
            raise ValueError("fixture initialization failed")
        if behavior == "blocked_initialize":
            time.sleep(60)

    client = _core.IpcClient(endpoint.detach(), initialize, process=True)

    async def serve() -> None:
        try:
            await client.serve(_ChildHost())
        finally:
            await client.aclose()

    asyncio.run(serve())
    os._exit(0)


async def _unused_entrypoint(context: Any) -> None:
    assert context.job.metadata == f"wire-{context.job.id}"
    context.shutdown("stress complete")


class _Executor(JobExecutor):
    def __init__(self, method: str, behavior: str = "normal") -> None:
        production_deadlines = behavior in {"real_job", "watchdog"}
        super().__init__(
            job_entrypoint_fnc=_unused_entrypoint,
            session_end_fnc=None,
            simulation_end_fnc=None,
            initialize_timeout=2
            if behavior == "blocked_initialize"
            else _PROCESS_INITIALIZE_TIMEOUT,
            close_timeout=_PROCESS_CLOSE_TIMEOUT if production_deadlines else 0.25,
            session_end_timeout=0.25,
            ping_interval=0.05,
            ping_timeout=60 if production_deadlines else 0.2,
            high_ping_threshold=0.1,
            http_proxy=None,
            mp_ctx=mp.get_context(method),
            loop=asyncio.get_running_loop(),
        )
        self.behavior = behavior
        self.launch_gate: asyncio.Event | None = None
        self.launch_entered = asyncio.Event()
        self.process_spawned = asyncio.Event()
        self.receipt_gate: asyncio.Event | None = None
        self.completion: ProcessCompletion | None = None

    async def _core_completed(self, completion: ProcessCompletion, pid: int) -> None:
        self.completion = completion
        await super()._core_completed(completion, pid)

    def _create_process(self, endpoint: socket.socket) -> Any:
        if self.behavior == "real_job":
            return super()._create_process(endpoint)
        return self._mp_ctx.Process(target=_ipc_child, args=(endpoint, self.behavior))

    async def _core_launch(self, endpoint: socket.socket) -> None:
        self.launch_entered.set()
        if self.launch_gate is not None:
            await self.launch_gate.wait()
        await super()._core_launch(endpoint)
        self.process_spawned.set()
        if self.receipt_gate is not None:
            await self.receipt_gate.wait()


@pytest.fixture
async def executor_factory(
    own_executor: Callable[[JobExecutor], JobExecutor],
) -> AsyncIterator[Callable[[str, str], _Executor]]:
    owned: list[_Executor] = []

    def create(method: str, behavior: str = "normal") -> _Executor:
        executor = _Executor(method, behavior)
        own_executor(executor)
        owned.append(executor)
        return executor

    try:
        yield create
    finally:
        for executor in owned:
            if executor.launch_gate is not None:
                executor.launch_gate.set()
            if executor.receipt_gate is not None:
                executor.receipt_gate.set()


async def _assert_reaped(executor: _Executor) -> None:
    await asyncio.wait_for(executor.join(), timeout=5)
    assert executor.exitcode is not None, "native owner must deliver the start method's receipt"
    pid = executor.pid
    assert pid is not None
    assert not psutil.pid_exists(pid), "a zombie is not a reaped process"
    if os.name == "posix" and executor._mp_ctx.get_start_method() == "spawn":
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


@pytest.mark.parametrize("method", _METHODS)
async def test_process_resource_lifecycle_matrix(
    method: str, executor_factory: Callable[[str, str], _Executor]
) -> None:
    executor = executor_factory(method, "normal")
    await executor.start()
    await executor.initialize()
    await asyncio.wait_for(executor.aclose(), timeout=3)
    await _assert_reaped(executor)
    assert executor.exitcode == 0


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("behavior", ["failed_initialize", "blocked_initialize"])
async def test_failed_initialization_retains_native_cleanup_ownership(
    method: str, behavior: str, executor_factory: Callable[[str, str], _Executor]
) -> None:
    executor = executor_factory(method, behavior)
    await executor.start()
    with pytest.raises((RuntimeError, TimeoutError, ConnectionError)):
        await executor.initialize()
    await asyncio.wait_for(executor.aclose(), timeout=3)
    await _assert_reaped(executor)
    assert executor.exitcode != 0


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("after_launch", [False, True])
async def test_cancelled_start_does_not_abandon_a_launched_process(
    method: str, after_launch: bool, executor_factory: Callable[[str, str], _Executor]
) -> None:
    executor = executor_factory(method, "normal")
    executor.launch_gate = asyncio.Event()
    executor.receipt_gate = asyncio.Event()
    starting = asyncio.create_task(executor.start())
    await executor.launch_entered.wait()
    if after_launch:
        executor.launch_gate.set()
        await asyncio.wait_for(executor.process_spawned.wait(), timeout=5)
    starting.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await starting
    executor.launch_gate.set()
    # In the postlaunch case, keep the host launch result blocked while native close reaps.
    if not after_launch:
        executor.receipt_gate.set()
    await asyncio.wait_for(executor.aclose(), timeout=5)
    await _assert_reaped(executor)
    executor.receipt_gate.set()


@pytest.mark.parametrize("method", _METHODS)
async def test_blocked_python_health_kill(
    method: str, executor_factory: Callable[[str, str], _Executor], tmp_path: Path
) -> None:
    executor = executor_factory(method, "normal")
    await executor.start()
    await executor.initialize()
    entered = tmp_path / "blocked-callback.entered"
    job = RunningJobInfo(
        JobAcceptArguments("", "", str(entered)), agent.Job(id="blocked-python"), "", "", "", True
    )
    started = time.monotonic()
    with pytest.raises((RuntimeError, ConnectionError)):
        await executor.launch_job(job)
    await _assert_reaped(executor)
    assert entered.exists(), "the real host loop must enter the blocking callback"
    # The configured 200 ms health deadline, 500 ms signal grace and bounded receipt wait
    # must finish well before the intentionally blocked 60-second host callback.
    assert time.monotonic() - started < 3
    assert executor.killed
    assert executor.exitcode != 0


async def test_stack_dump_off_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import tempfile
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from livekit.agents.ipc import child

    entered = threading.Event()
    release = threading.Event()
    watchdog_fired = threading.Event()
    original_file = tempfile.NamedTemporaryFile

    def stalled_file(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        if not release.wait(timeout=2):
            watchdog_fired.set()
        return original_file(*args, **kwargs)

    # The bounded writer is already warm when production creates the job loop.
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stack-dump-fixture")
    writer.submit(lambda: None).result()
    monkeypatch.setenv("LK_DUMP_STACK_TRACES", "1")
    monkeypatch.setenv("LK_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(child, "_stack_dump_worker", writer, raising=False)
    monkeypatch.setattr(child, "_stack_dump_busy", threading.Lock(), raising=False)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", stalled_file)
    try:
        child._dump_stack_traces()

        async def filesystem_entered() -> None:
            while not entered.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(filesystem_entered(), timeout=3)
        # A tick must run while the actual writer's file-open remains blocked.
        await asyncio.sleep(0)
        assert not watchdog_fired.is_set(), "the dump's filesystem wait blocked the job loop"
        assert not release.is_set()
    finally:
        release.set()
        await asyncio.to_thread(writer.shutdown, wait=True)
    assert list(tmp_path.glob("*.stacktrace"))


@pytest.mark.parametrize("method", _METHODS)
async def test_launch_failure_keeps_its_cause_and_settles_close(
    method: str,
    executor_factory: Callable[[str, str], _Executor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = executor_factory(method, "normal")
    # Multiprocessing must reject the unpicklable target before creating a child.
    monkeypatch.setattr(
        executor,
        "_create_process",
        lambda endpoint: executor._mp_ctx.Process(target=lambda: None),
    )
    with pytest.raises(RuntimeError, match="pickle|local object"):
        await asyncio.wait_for(executor.start(), timeout=5)
    await asyncio.wait_for(executor.aclose(), timeout=3)
    await asyncio.wait_for(executor.join(), timeout=3)
    assert executor.pid is None


@pytest.mark.parametrize("method", _METHODS)
async def test_post_spawn_failure_keeps_native_owner_and_original_cause(
    method: str,
    executor_factory: Callable[[str, str], _Executor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    suffix = (
        "spawn_win32"
        if os.name == "nt"
        else "forkserver"
        if method == "forkserver"
        else "spawn_posix"
    )
    popen = importlib.import_module(f"multiprocessing.popen_{suffix}").Popen
    initialize = popen.__init__

    def fail_after_spawn(self: Any, process: Any) -> None:
        initialize(self, process)
        raise OSError("fixture bootstrap failed after OS launch")

    monkeypatch.setattr(popen, "__init__", fail_after_spawn)
    executor = executor_factory(method, "normal")
    with pytest.raises(RuntimeError, match="fixture bootstrap failed after OS launch"):
        await asyncio.wait_for(executor.start(), timeout=5)
    assert executor.pid is not None, "partial Popen failure must still publish native ownership"
    await asyncio.wait_for(executor.aclose(), timeout=3)
    await _assert_reaped(executor)
    assert executor.exitcode != 0


@pytest.mark.parametrize("method", _METHODS)
async def test_ready_pool_processes_use_native_exit_and_reaping_handles(
    method: str,
    executor_factory: Callable[[str, str], _Executor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback_must_not_run() -> bool:
        raise AssertionError("ready process observation crossed the Python loop")

    class Host:
        def __init__(self) -> None:
            self.ready: list[_Executor] = []

        def _core_create(self) -> _Executor:
            executor = executor_factory(method, "normal")
            monkeypatch.setattr(executor, "_core_wait_exit", callback_must_not_run)
            monkeypatch.setattr(executor, "_core_join", callback_must_not_run)
            return executor

        def emit(self, event: str, executor: _Executor) -> None:
            if event == "process_ready":
                self.ready.append(executor)

    host = Host()
    pool = _core.ProcPool(
        host,
        num_idle_processes=1,
        initialize_timeout=_PROCESS_INITIALIZE_TIMEOUT,
        close_timeout=0.25,
    )
    try:
        await asyncio.wait_for(pool.start(), timeout=_PROCESS_INITIALIZE_TIMEOUT + 5)
        executor = pool.processes[0]
        pid = executor.pid
        assert pid is not None
        _kill_pid(pid)
        # Native ownership/eligibility keeps progressing while host callbacks cannot run.
        time.sleep(1)

        async def replacement_ready() -> None:
            while not any(replacement is not executor for replacement in host.ready):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(replacement_ready(), timeout=5)
        assert executor not in pool.processes
        assert any(
            replacement.pid != pid for replacement in host.ready if replacement is not executor
        )
    finally:
        await asyncio.wait_for(pool.aclose(), timeout=5)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize(
    "fault", ["wire_malformed", "wire_truncated", "wire_oversized_prefix", "wire_eof"]
)
async def test_real_ipc_protocol_failure_vectors(
    method: str, fault: str, executor_factory: Callable[[str, str], _Executor]
) -> None:
    from livekit.agents.ipc import JobStatus

    executor = executor_factory(method, fault)
    await executor.start()
    with pytest.raises((RuntimeError, ConnectionError)):
        await asyncio.wait_for(
            executor.initialize(), timeout=_PROCESS_INITIALIZE_TIMEOUT + _PROCESS_CLOSE_TIMEOUT
        )
    await _assert_reaped(executor)
    assert executor.status == JobStatus.FAILED
    assert executor.completion is not None
    failure = executor.completion.failure
    if fault == "wire_oversized_prefix":
        assert isinstance(failure, IpcFailureTooLarge)
        assert failure.size == 2**32 - 1
    elif fault == "wire_malformed":
        assert isinstance(failure, IpcFailureMalformed)
    else:
        assert isinstance(failure, IpcFailureIo)
        assert failure.kind == "UnexpectedEof"
    # The peer deliberately returned zero after sending malformed/truncated data.
    if fault != "wire_oversized_prefix":
        assert executor.exitcode == 0


@pytest.mark.parametrize("method", _METHODS)
async def test_real_ipc_stress_matrix(
    method: str, own_executor: Callable[[JobExecutor], JobExecutor], tmp_path: Path
) -> None:
    from tests.conformance.ipc_processes import process_snapshot

    parent = psutil.Process()
    retained: list[weakref.ReferenceType[JobExecutor]] = []

    async def cycle(index: int) -> None:
        executor = _Executor(method, "real_job")
        own_executor(executor)
        retained.append(weakref.ref(executor))
        started_at = time.monotonic()
        phases: dict[str, float] = {}
        try:
            await executor.start()
            phases["launch"] = time.monotonic() - started_at
            await executor.initialize()
            phases["ready"] = time.monotonic() - started_at
            info = RunningJobInfo(
                JobAcceptArguments("", "", ""),
                agent.Job(id=f"stress-{index}", metadata=f"wire-stress-{index}"),
                "",
                "",
                "",
                True,
            )
            await executor.launch_job(info)
            phases["job_accepted"] = time.monotonic() - started_at
            await asyncio.wait_for(executor._core_wait_exit(), timeout=5)
            await _assert_reaped(executor)
            assert executor.completion is not None
            assert executor.completion.failure is None
            assert executor.exitcode == 0
        finally:
            await executor.aclose()
            phases["closed"] = time.monotonic() - started_at
            (tmp_path / f"ipc-{method}-{index}-phases.json").write_text(
                json.dumps(phases, indent=2)
            )

    async def batch(index: int) -> None:
        async with asyncio.TaskGroup() as jobs:
            for offset in range(_CONCURRENT_JOBS):
                jobs.create_task(cycle(index * _CONCURRENT_JOBS + offset))
        deadline = time.monotonic() + 5
        while True:
            await asyncio.sleep(0)
            gc.collect()
            if not any(reference() is not None for reference in retained):
                break
            assert time.monotonic() < deadline, "settled jobs retained their host references"
            await asyncio.sleep(0.01)

    # Include the existing executor's declared cache capacity, not only today's idle occupancy.
    import threading

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: None)
    executor = loop._default_executor
    assert isinstance(executor, ThreadPoolExecutor)
    executor_capacity = executor._max_workers
    launch_workers = threading.Barrier(executor_capacity)
    await asyncio.gather(
        *[
            loop.run_in_executor(None, launch_workers.wait, _PROCESS_INITIALIZE_TIMEOUT)
            for _ in range(executor_capacity)
        ]
    )
    await batch(0)
    await asyncio.sleep(_NATIVE_CACHE_IDLE_TIMEOUT)
    baseline = process_snapshot(parent)
    baseline_children = {child.pid for child in parent.children(recursive=True)}
    baseline_tasks = len(asyncio.all_tasks())
    observations: list[dict[str, Any]] = []
    batches = int(os.environ.get("LIVEKIT_IPC_SOAK_BATCHES", "3"))
    assert batches > 0
    for index in range(1, batches + 1):
        await batch(index)
        peak_after_close = process_snapshot(parent)
        settling_started = time.monotonic()
        deadline = settling_started + _PROCESS_CLOSE_TIMEOUT + _NATIVE_CACHE_IDLE_TIMEOUT
        while parent.num_threads() > baseline["threads"] and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        current = process_snapshot(parent)
        children = {child.pid for child in parent.children(recursive=True)}
        observed = {
            "batch": index,
            "resources": current,
            "peak_after_close": peak_after_close,
            "cache_settlement_seconds": time.monotonic() - settling_started,
            "children": sorted(children),
            "tasks": len(asyncio.all_tasks()),
            "host_references": sum(ref() is not None for ref in retained),
        }
        observations.append(observed)
        artifact = tmp_path / f"ipc-stress-{method}.json"
        artifact.write_text(
            json.dumps(
                {
                    "method": method,
                    "concurrent_jobs": _CONCURRENT_JOBS,
                    "cache_idle_seconds": _NATIVE_CACHE_IDLE_TIMEOUT,
                    "python_executor_capacity": executor_capacity,
                    "baseline": baseline,
                    "batches": observations,
                },
                indent=2,
            )
        )
        assert not any(ref() is not None for ref in retained), (
            f"host references retained: {artifact}"
        )
        assert children == baseline_children, f"process/helper retention: {artifact}"
        assert current["handles"] <= baseline["handles"], f"descriptor retention: {artifact}"
        assert current["threads"] <= baseline["threads"], f"thread retention: {artifact}"
        assert observed["tasks"] == baseline_tasks, f"async task retention: {artifact}"


async def test_ipc_fixture_watchdog_failure(
    own_executor: Callable[[JobExecutor], JobExecutor],
    tmp_path: Path,
) -> None:
    if os.environ.get("LIVEKIT_IPC_WATCHDOG_CHILD") == "1":

        class StuckCloseExecutor(_Executor):
            async def aclose(self) -> None:
                await asyncio.sleep(60)

        executor = StuckCloseExecutor("spawn", "watchdog")
        own_executor(executor)
        await executor.start()
        await executor.initialize()
        assert executor.pid is not None
        assert psutil.Process(executor.pid).is_running()

        return  # teardown must fail, capture evidence, kill, and reap the real child

    from tests.conformance.ipc_processes import run_owned_process

    artifact_dir = tmp_path / "watchdog-child"
    result = await asyncio.to_thread(
        run_owned_process,
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).resolve()) + "::test_ipc_fixture_watchdog_failure",
            "--e2e",
            "--basetemp",
            str(artifact_dir),
            "-q",
        ],
        tmp_path / "watchdog-run",
        {**os.environ, "LIVEKIT_IPC_WATCHDOG_CHILD": "1"},
        30,
    )
    assert result.returncode != 0, result.stdout + (result.stderr or "")
    assert "missed cleanup deadline" in result.stdout + (result.stderr or "")
    artifacts = list(artifact_dir.rglob("executor-*.json"))
    assert artifacts, result.stdout + (result.stderr or "")
    for artifact in artifacts:
        evidence = json.loads(artifact.read_text())
        assert evidence["failure"] is not None
        assert evidence["before"], "capture the owned tree before killing it"
        assert evidence["after"] == [], "emergency cleanup must reap rather than accept zombies"


def test_process_watchdog_preserves_failure_after_cleaning_the_owned_tree(tmp_path: Path) -> None:
    import subprocess

    from tests.conformance.ipc_processes import run_owned_process

    result = run_owned_process(
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path / "nonzero",
        dict(os.environ),
        5,
    )
    assert result.returncode == 7
    script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    artifact_dir = tmp_path / "timed-out"
    with pytest.raises(AssertionError, match="watchdog failed") as failed:
        run_owned_process([sys.executable, "-c", script], artifact_dir, dict(os.environ), 0.25)
    assert isinstance(failed.value.__cause__, subprocess.TimeoutExpired)
    evidence = json.loads((artifact_dir / "processes.json").read_text())
    assert len(evidence["before"]) == 2, "capture both the parent and its inherited child"
    assert evidence["after"] == [], "cleanup must not accept zombies as reaped"
