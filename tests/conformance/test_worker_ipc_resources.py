"""Real multiprocessing launch, native health supervision and independent reaping receipts."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import psutil
import pytest

from livekit.agents import _core
from livekit.agents.ipc import JobExecutor
from livekit.agents.job import JobAcceptArguments, RunningJobInfo
from livekit.protocol import agent

pytestmark = pytest.mark.e2e

_METHODS = ["spawn", *(["forkserver"] if sys.platform == "linux" else [])]


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        psutil.Process(pid).kill()
    else:
        os.kill(pid, signal.SIGKILL)


class _ChildHost:
    async def _core_start_job(self, fields: dict[str, Any]) -> None:
        # Deliberately block the Python event loop so no ping callback can run.
        time.sleep(60)

    async def _core_dump_stacks(self) -> None:
        pass


def _ipc_child(endpoint: socket.socket, behavior: str) -> None:
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: None)
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


async def _unused_entrypoint(context: Any) -> None:
    pass


class _Executor(JobExecutor):
    def __init__(self, method: str, behavior: str = "normal") -> None:
        super().__init__(
            job_entrypoint_fnc=_unused_entrypoint,
            session_end_fnc=None,
            simulation_end_fnc=None,
            initialize_timeout=2,
            close_timeout=0.25,
            session_end_timeout=0.25,
            ping_interval=0.05,
            ping_timeout=0.2,
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

    def _create_process(self, endpoint: socket.socket) -> Any:
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
async def executor_factory() -> AsyncIterator[Callable[[str, str], _Executor]]:
    owned: list[_Executor] = []

    def create(method: str, behavior: str = "normal") -> _Executor:
        executor = _Executor(method, behavior)
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
            try:
                await asyncio.wait_for(executor.aclose(), timeout=5)
                if executor.started:
                    await asyncio.wait_for(executor.join(), timeout=5)
            except BaseException:
                # Capture the failed fixture's evidence before emergency tree cleanup.
                pid = executor.pid
                evidence: list[tuple[int, str]] = []
                if pid is not None:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        process = psutil.Process(pid)
                        evidence = [
                            (child.pid, child.status())
                            for child in process.children(recursive=True)
                        ]
                        evidence.insert(0, (pid, process.status()))
                    for child_pid, _ in reversed(evidence):
                        with contextlib.suppress(ProcessLookupError, psutil.NoSuchProcess):
                            _kill_pid(child_pid)
                cleanup_error: str | None = None
                if executor.started:
                    try:
                        await asyncio.wait_for(executor.join(), timeout=5)
                    except BaseException as error:
                        cleanup_error = repr(error)

                async def owned_tree_gone(evidence: list[tuple[int, str]] = evidence) -> None:
                    while any(psutil.pid_exists(owned_pid) for owned_pid, _ in evidence):
                        await asyncio.sleep(0.01)

                try:
                    await asyncio.wait_for(owned_tree_gone(), timeout=5)
                except TimeoutError:
                    cleanup_error = (
                        f"{cleanup_error}; owned processes remained after emergency cleanup"
                    )
                pytest.fail(
                    f"native process fixture did not settle: {evidence}; cleanup={cleanup_error}"
                )


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
    method: str, executor_factory: Callable[[str, str], _Executor]
) -> None:
    executor = executor_factory(method, "normal")
    await executor.start()
    await executor.initialize()
    job = RunningJobInfo(
        JobAcceptArguments("", "", ""), agent.Job(id="blocked-python"), "", "", "", True
    )
    await executor.launch_job(job)
    started = time.monotonic()
    await _assert_reaped(executor)
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
    pool = _core.ProcPool(host, num_idle_processes=1, initialize_timeout=2, close_timeout=0.25)
    try:
        await asyncio.wait_for(pool.start(), timeout=5)
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
