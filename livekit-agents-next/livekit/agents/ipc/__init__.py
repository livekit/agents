"""The worker's job processes, as the core (``_core.JobSupervisor``, ``_core.ProcPool``)
supervises and pools them: this module starts them (``multiprocessing``, the forkserver's
preloads, a job thread), finishes the start method's bookkeeping and hands the core the user's
callbacks. What runs inside a process is ``ipc.child``."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import socket
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any, Literal, Protocol, cast

from livekit.agents._generated.worker_ipc import (
    JobRunnerOptions as JobRunnerOptions,
    JobStatus as JobStatus,
)

from .. import _core, utils
from .._generated.worker_ipc import JobCompletion
from .._generated.worker_server import RunningJobInfo
from ..job import JobContext
from ..log import logger
from . import child, prometheus


class _JobThread:
    """A job thread: it reports its return on its launch, it is never killed."""

    def __init__(self, args: child.JobArgs, launch: _core.JobLaunch) -> None:
        def run() -> None:
            try:
                child.thread_main(args)
            finally:
                launch.thread_exited()

        self._thread = threading.Thread(target=run, name="job_thread_runner")
        self.native_id: int | None = None

    def start(self) -> None:
        self._thread.start()
        self.native_id = self._thread.native_id


class JobLauncher:
    """Starts one job runner for its ``_core.JobSupervisor``: a process of ``mp_ctx``'s start
    method (``spawn``, or ``forkserver`` on Linux) or a thread of this process, on the job's end
    of the channel the core made; ``user_arguments`` is the job's ``JobProcess.user_arguments``.
    The core owns everything after: the runner's state, its
    supervision and its end. The ``_core_*`` methods are the core's calls."""

    def __init__(
        self,
        *,
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        session_end_timeout: float,
        preload_modules: Sequence[str] = (),
        env: Sequence[tuple[str, str]] = (),
        mp_ctx: BaseContext | None = None,
        user_arguments: Any | None = None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._mp_ctx = mp_ctx
        self._proc: Any = None
        self._args = child.JobArgs(
            job_entrypoint_fnc=job_entrypoint_fnc,
            session_end_fnc=session_end_fnc,
            simulation_end_fnc=simulation_end_fnc,
            session_end_timeout=session_end_timeout,
            preload_modules=preload_module_names(preload_modules),
            user_arguments=user_arguments,
            env=tuple(env),
        )

    async def _core_launch(self, endpoint: int, launch: _core.JobLaunch) -> None:
        # the job's end: a socket on Unix, a named pipe on Windows
        child_end = child.PipeEnd(endpoint) if os.name == "nt" else socket.socket(fileno=endpoint)
        if launch.thread:
            try:
                # the thread shares this process's logging and owns the job's end
                self._proc = _JobThread(
                    child.JobArgs(**{**vars(self._args), "mp_cch": child_end}), launch
                )
                self._proc.start()
            except BaseException:
                child_end.close()
                raise
            return
        try:
            if self._mp_ctx is None:
                raise ValueError("a job process needs a multiprocessing context")
            self._proc = self._create_process(self._mp_ctx, child_end)
            method = self._mp_ctx.get_start_method()

            def register(receipt: ProcessReceipt) -> None:
                launch.process_started(receipt.pid, receipt.sentinel, method, receipt.owned_handles)

            # The parent must keep receiving shutdown signals while children start.
            # Job processes install their own signal policy in ipc.child.
            await self._loop.run_in_executor(None, start_process, self._proc, method, register)
        finally:
            # the process got its own copy of the end
            child_end.close()

    async def _core_completed(self, completion: JobCompletion) -> None:
        if isinstance(self._proc, BaseProcess):
            # Start-method bookkeeping only: the native owner has already reaped the child.
            self._proc._popen.returncode = completion.exit_code  # type: ignore[attr-defined]
            self._proc._popen.finalizer()  # type: ignore[attr-defined]

    def _create_process(
        self, mp_ctx: BaseContext, cch: socket.socket | child.PipeEnd
    ) -> BaseProcess:
        from ..cli import _log_config

        levels = {"root": logging.getLogger().level}
        for log in logging.Logger.manager.loggerDict.values():
            if isinstance(log, logging.Logger):
                levels[log.name] = log.level
        self._args.mp_cch = cch
        self._args.log_config = _log_config
        self._args.logger_levels = levels
        return cast(
            BaseProcess,
            mp_ctx.Process(  # type: ignore[attr-defined]
                target=child.job_main, args=(self._args,), name="job_proc"
            ),
        )

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None:
        if isinstance(self._proc, _JobThread):
            extra = {**extra, "tid": self._proc.native_id}
        logger.log(level, msg, extra=extra, exc_info=exc)


class ProcPool(
    utils.EventEmitter[
        Literal[
            "process_created",
            "process_started",
            "process_ready",
            "process_closed",
            "process_job_launched",
        ]
    ]
):
    """The warm job processes. The core keeps ``num_idle_processes`` warm, hands a job the next
    one, retries a failed spawn or launch and owns each process's state
    (``_core.ProcPool``, ``_core.JobSupervisor``); this class makes the launchers and turns the
    core's notifications into events and Prometheus metrics."""

    def __init__(
        self,
        *,
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        num_idle_processes: int,
        options: JobRunnerOptions,
        session_end_timeout: float,
        preload_modules: Sequence[str] = (),
        mp_ctx: BaseContext,
        env: Sequence[tuple[str, str]] = (),
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._launcher_args: dict[str, Any] = {
            "job_entrypoint_fnc": job_entrypoint_fnc,
            "session_end_fnc": session_end_fnc,
            "simulation_end_fnc": simulation_end_fnc,
            "session_end_timeout": session_end_timeout,
            "preload_modules": preload_module_names(preload_modules),
            "env": env,
            "mp_ctx": mp_ctx,
            "loop": loop,
        }
        self._pool = _core.ProcPool(self, num_idle_processes=num_idle_processes, options=options)

    @property
    def processes(self) -> list[_core.JobSupervisor]:
        return self._pool.processes

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return self._pool.active_jobs

    @property
    def _jobs_waiting_for_process(self) -> int:
        return self._pool.jobs_waiting

    def get_by_job_id(self, job_id: str) -> _core.JobSupervisor | None:
        return self._pool.get_by_job_id(job_id)

    async def start(self) -> None:
        await self._pool.start()

    async def aclose(self) -> None:
        await self._pool.aclose()

    async def join_jobs(self) -> None:
        """Waits until every process a job was launched on exited (the drain)."""
        await self._pool.join_jobs()

    async def launch_job(self, info: RunningJobInfo) -> None:
        await self._pool.launch_job(info)

    def set_target_idle_processes(self, num_idle_processes: int) -> None:
        self._pool.target_idle_processes = num_idle_processes

    @property
    def target_idle_processes(self) -> int:
        return self._pool.target_idle_processes

    def emit(self, event: Any, *args: Any) -> None:
        """The core's one notification of a process's step."""
        prometheus.process_event(event, *args)
        super().emit(event, *args)

    async def _core_create(self) -> JobLauncher:
        return JobLauncher(**self._launcher_args)


_FORKSERVER_PRELOAD_MODULES = "LIVEKIT_AGENTS_FORKSERVER_PRELOAD_MODULES"


class _ForkServerState(Protocol):
    # CPython publishes these launch-integration fields, but typeshed omits them.
    _forkserver_pid: int | None
    _preload_modules: list[str]


def preload_module_names(modules: Sequence[str]) -> tuple[str, ...]:
    """Validate the launch configuration before allocating a child process."""
    if isinstance(modules, str):
        raise TypeError("preload_modules must be a sequence of module names, not a string")
    names = tuple(modules)
    for module in names:
        if not isinstance(module, str) or not all(
            part.isidentifier() for part in module.split(".")
        ):
            raise ValueError(f"invalid preload module name {module!r}")
    return names


def load_preload_modules(modules: Sequence[str]) -> None:
    """Import in order and retain the failing module and original cause."""
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            # multiprocessing's forkserver ignores ImportError from its preload list.
            raise RuntimeError(f"preload module {module!r} failed: {error}") from error


def configure_forkserver(context: BaseContext, modules: Sequence[str]) -> None:
    """Install one fail-closed preload entrypoint before the forkserver starts."""
    modules = preload_module_names(modules)
    from multiprocessing.forkserver import _forkserver

    state = cast(_ForkServerState, _forkserver)
    # Retain the ordered configuration in multiprocessing's own preload inventory. The
    # bootstrap already imports these modules, so its trailing imports are cached no-ops.
    inventory = ["livekit.agents.ipc._preload_forkserver", *modules]
    if state._forkserver_pid is not None:
        if state._preload_modules != inventory:
            raise RuntimeError(
                "the running forkserver has different preload_modules; configure all servers "
                "with the same ordered preloads before launching jobs, or use spawn"
            )
        return
    # The forkserver is a fresh interpreter; only its environment survives that exec.
    os.environ[_FORKSERVER_PRELOAD_MODULES] = os.pathsep.join(modules)
    context.set_forkserver_preload(inventory)


@dataclass(frozen=True)
class ProcessReceipt:
    pid: int
    sentinel: int
    owned_handles: list[int]


def start_process(
    process: BaseProcess, method: str, register: Callable[[ProcessReceipt], None]
) -> None:
    if sys.platform == "win32":
        from multiprocessing.popen_spawn_win32 import Popen
    elif method == "forkserver":
        from multiprocessing.popen_forkserver import Popen
    else:
        # Both POSIX concrete classes have the same bootstrap contract.
        from multiprocessing.popen_spawn_posix import Popen  # type: ignore[assignment]

    class OwnedPopen(Popen):
        def __init__(self, process_obj: BaseProcess) -> None:
            try:
                super().__init__(process_obj)
            finally:
                # Upstream may fail after OS launch but before BaseProcess.start assigns Popen.
                # No asynchronous hop or descriptor allocation can intervene in this handoff.
                try:
                    sentinel = self.sentinel
                    finalizer = self.finalizer
                except AttributeError:
                    pass  # No process identity was published by this start method.
                else:
                    try:
                        pid = self.pid
                    except AttributeError:
                        if method != "forkserver":
                            raise
                        pid = 0
                    process_obj._popen = self  # type: ignore[attr-defined]
                    handles: Sequence[int] = finalizer._args  # type: ignore[attr-defined]
                    owned = [handle for handle in handles if handle != sentinel]
                    finalizer.cancel()
                    register(ProcessReceipt(pid, sentinel, owned))

    def factory(process_obj: BaseProcess) -> OwnedPopen:
        # The temporary factory contains a native handle and must never enter the child pickle.
        del process_obj._Popen  # type: ignore[attr-defined]
        return OwnedPopen(process_obj)

    process._Popen = factory  # type: ignore[attr-defined]
    try:
        process.start()
    finally:
        # Native supervision is the only waiter, including when Popen.__init__ raised.
        from multiprocessing.process import _children  # type: ignore[attr-defined]

        _children.discard(process)


__all__ = [
    "JobLauncher",
    "JobRunnerOptions",
    "JobStatus",
    "ProcPool",
]
