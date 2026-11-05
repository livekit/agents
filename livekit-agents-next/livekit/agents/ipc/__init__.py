"""The worker's job processes, as the core (``_core.JobSupervisor``,
``_core.ProcPool``) supervises and pools them: this module spawns them (``multiprocessing``, the
forkserver's preloads), samples their memory (psutil) and hands the core the user's callbacks.
What runs inside a process is ``ipc.child``."""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing as mp
import os
import socket
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any, Literal, Protocol, cast

from livekit.agents._generated.worker_ipc import JobStatus as JobStatus

from .. import _core, utils
from .._generated.worker_ipc import JobCompletion, StartJobFields
from ..job import JobContext, JobExecutorType, RunningJobInfo
from ..log import logger
from ..utils import shortuuid
from . import child, prometheus


class _JobThread:
    """A job thread: it tells the core when it returned, it is never killed."""

    def __init__(self, args: child.JobArgs, exited: Callable[[], None]) -> None:
        def run() -> None:
            try:
                child.thread_main(args)
            finally:
                exited()

        self._thread = threading.Thread(target=run, name="job_thread_runner")
        self.native_id: int | None = None

    def start(self) -> None:
        self._thread.start()
        self.native_id = self._thread.native_id


class JobExecutor:
    """One job's process, or (``thread``) a thread of this process, as the core supervises it
    (``_core.JobSupervisor``): its initialization, pings, memory and close, until it exited. A
    thread cannot be killed (a late initialization, ack or exit is reported and waited for), has no
    pong deadline nor memory limit, and is not in the metrics. The ``_core_*`` methods are the
    core's calls."""

    _pch: _core.JobSupervisor

    def __init__(
        self,
        *,
        thread: bool = False,
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        initialize_timeout: float,
        close_timeout: float,
        session_end_timeout: float,
        preload_modules: Sequence[str] = (),
        memory_warn_mb: float = 0,
        memory_limit_mb: float = 0,
        ping_interval: float,
        ping_timeout: float = 0,
        high_ping_threshold: float,
        http_proxy: str | None,
        env: Sequence[tuple[str, str]] = (),
        mp_ctx: BaseContext | None = None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        preload_modules = preload_module_names(preload_modules)
        self._loop = loop
        self._mp_ctx = mp_ctx or mp.get_context(
            "forkserver" if sys.platform == "linux" else "spawn"
        )
        if not thread and self._mp_ctx.get_start_method() not in ("spawn", "forkserver"):
            raise ValueError("job processes support the spawn and Linux forkserver start methods")
        if (
            not thread
            and self._mp_ctx.get_start_method() == "forkserver"
            and sys.platform != "linux"
        ):
            logger.warning("the forkserver is used on Linux only, job processes are spawned")
            self._mp_ctx = mp.get_context("spawn")
        self._thread = thread
        # the core's supervision of the process; a thread has no pong deadline (it cannot be
        # killed for being late) nor memory limit
        self._opts = {
            "initialize_timeout": initialize_timeout,
            "close_timeout": close_timeout,
            "memory_warn_mb": 0 if thread else memory_warn_mb,
            "memory_limit_mb": 0 if thread else memory_limit_mb,
            "ping_interval": ping_interval,
            "ping_timeout": 0 if thread else ping_timeout,
            "high_ping_threshold": high_ping_threshold,
        }
        self._http_proxy = http_proxy
        self._exitcode: int | None = None
        self._pid: int | None = None
        self._start_atask: asyncio.Task[None] | None = None
        self._launched: asyncio.Future[None] = loop.create_future()
        # the core's run of the process, from its initialization to its exit
        self._supervise_atask: asyncio.Future[bool] | None = None
        self._closing = False
        self.user_arguments: Any | None = None
        self._job_status: JobStatus | None = None
        self._running_job: RunningJobInfo | None = None
        self._args = child.JobArgs(
            job_entrypoint_fnc=job_entrypoint_fnc,
            session_end_fnc=session_end_fnc,
            simulation_end_fnc=simulation_end_fnc,
            session_end_timeout=session_end_timeout,
            preload_modules=tuple(preload_modules),
            env=tuple(env),
        )
        self._id = shortuuid("THEXEC_" if thread else "PCEXEC_")

    @property
    def id(self) -> str:
        return self._id

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    @property
    def killed(self) -> bool:
        return self.started and self._pch.killed

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def started(self) -> bool:
        return self._supervise_atask is not None

    @property
    def status(self) -> JobStatus:
        if self._job_status is None:
            raise RuntimeError("job status not available")
        return self._job_status

    @property
    def running_job(self) -> RunningJobInfo | None:
        return self._running_job

    async def start(self) -> None:
        """start the process"""
        if self._start_atask is not None:
            raise RuntimeError("process already started")
        if self._closing:
            raise RuntimeError("process is closed")
        # keep a handle so aclose() can await an abandoned (cancelled) start()
        self._start_atask = asyncio.create_task(self._start())
        await asyncio.shield(self._start_atask)

    async def _core_start(self) -> _core.JobSupervisor:
        await self.start()
        return self._pch

    async def _start(self) -> None:
        if self._closing:
            raise RuntimeError("process is closed")
        # The core launches through _core_launch; its supervision survives caller loss.
        self._pch = _core.JobSupervisor(self, thread=self._thread, **self._opts)
        self._supervise_atask = asyncio.ensure_future(self._pch.run())
        launched_or_over: set[asyncio.Future[Any]] = {self._launched, self._supervise_atask}
        await asyncio.wait(launched_or_over, return_when=asyncio.FIRST_COMPLETED)
        if not self._launched.done():
            raise RuntimeError("the job runner could not be launched")
        self._launched.result()

    async def _core_launch(self, endpoint: int) -> None:
        # the job's end: a socket on Unix, a named pipe on Windows
        child_end = child.PipeEnd(endpoint) if os.name == "nt" else socket.socket(fileno=endpoint)
        method = self._mp_ctx.get_start_method()

        def register(receipt: ProcessReceipt) -> None:
            self._pid = receipt.pid or None
            self._pch._core_launched_process(
                receipt.pid, receipt.sentinel, method, receipt.owned_handles
            )

        try:
            self._proc = self._create_process(child_end)
            if self._thread:
                self._proc.start()
            else:
                # The parent must keep receiving shutdown signals while children start.
                # Job processes install their own signal policy in ipc.child.
                await self._loop.run_in_executor(None, start_process, self._proc, method, register)
        except BaseException as error:
            if self._thread:
                child_end.close()
            if not self._launched.done():
                self._launched.set_exception(error)
            raise
        finally:
            # a process got its own copy of the end
            if not self._thread:
                child_end.close()
        if not self._launched.done():
            self._launched.set_result(None)

    async def _core_completed(self, completion: JobCompletion) -> None:
        exitcode = completion.exit_code
        self._exitcode = exitcode
        if not self._thread:
            # Start-method bookkeeping only: the native owner has already reaped the child.
            self._proc._popen.returncode = exitcode
            self._proc._popen.finalizer()
        if self._running_job and not self._thread:
            prometheus.job_ended()
        self._job_status = (
            JobStatus.SUCCESS if exitcode == 0 and completion.failure is None else JobStatus.FAILED
        )

    def _create_process(self, cch: socket.socket | child.PipeEnd) -> Any:
        from ..cli import _log_config

        self._args.user_arguments = self.user_arguments
        if self._thread:
            # the thread shares this process's logging and owns the job's end
            return _JobThread(
                child.JobArgs(**{**vars(self._args), "mp_cch": cch}), self._pch._core_thread_exited
            )
        levels = {"root": logging.getLogger().level}
        for log in logging.Logger.manager.loggerDict.values():
            if isinstance(log, logging.Logger):
                levels[log.name] = log.level
        self._args.mp_cch = cch
        self._args.log_config = _log_config
        self._args.logger_levels = levels
        return self._mp_ctx.Process(  # type: ignore[attr-defined]
            target=child.job_main, args=(self._args,), name="job_proc"
        )

    async def join(self) -> None:
        """wait for the process to finish"""
        if self._supervise_atask is None:
            raise RuntimeError("process not started")
        await self._pch._core_join()

    async def _core_wait_exit(self) -> bool:
        if self._supervise_atask is None:
            return False
        return await self._pch._core_wait_exit()

    async def _core_join(self) -> bool:
        if self._supervise_atask is None:
            return False
        return await self._pch._core_join()

    async def initialize(self) -> None:
        """initialize the process: the core sends it its ``Hello`` and waits for it to be ready,
        killing it when late (``asyncio.TimeoutError``)"""
        elapsed_time = await self._pch.initialize(
            asyncio_debug=self._loop.get_debug(), http_proxy=self._http_proxy
        )
        if not self._thread:
            prometheus.proc_initialized(time_elapsed=elapsed_time)

    async def _core_launch_job(self, info: RunningJobInfo, launched_at: float) -> None:
        info.launched_at = launched_at
        await self.launch_job(info)

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")
        if not self.started or not self._pch.initialized:
            raise RuntimeError("process not initialized")
        await self._pch.start_job(
            StartJobFields(
                job=info.job.SerializeToString(),
                name=info.accept_arguments.name,
                identity=info.accept_arguments.identity,
                metadata=info.accept_arguments.metadata,
                url=info.url,
                token=info.token,
                worker_id=info.worker_id,
                fake_job=info.fake_job,
                received_at=info.received_at,
                accepted_at=info.accepted_at,
                assigned_at=info.assigned_at,
                launched_at=info.launched_at,
            )
        )
        if not self._thread:
            prometheus.job_started()
        self._job_status = JobStatus.RUNNING
        self._running_job = info

    async def aclose(self) -> None:
        """attempt to gracefully close the process: an initialized one is asked to shut down
        (and killed when late), any other is killed"""
        self._closing = True
        if self._supervise_atask is None:
            return
        await self._pch.aclose()

    async def kill(self) -> None:
        """forcefully kill the process"""
        if self._supervise_atask is None:
            raise RuntimeError("process not started")
        self._closing = True
        await self._pch.kill()

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None:
        logger.log(level, msg, extra={**extra, **self.logging_extra()}, exc_info=exc)

    def logging_extra(self) -> dict[str, Any]:
        if self._thread:
            proc = getattr(self, "_proc", None)
            extra: dict[str, Any] = {"tid": proc.native_id if proc else None}
        else:
            extra = {"pid": self.pid}
        if job := self._running_job:
            extra.update(job_id=job.job.id, room=job.job.room.name, room_id=job.job.room.sid)
        return extra


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
    one and retries a failed spawn or launch (``_core.ProcPool``); this class makes the
    executors."""

    def __init__(
        self,
        *,
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
        session_end_timeout: float,
        preload_modules: Sequence[str] = (),
        job_executor_type: JobExecutorType,
        mp_ctx: BaseContext,
        memory_warn_mb: float,
        memory_limit_mb: float,
        http_proxy: str | None,
        env: Sequence[tuple[str, str]] = (),
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._executor_args: dict[str, Any] = {
            "thread": job_executor_type == JobExecutorType.THREAD,
            "job_entrypoint_fnc": job_entrypoint_fnc,
            "session_end_fnc": session_end_fnc,
            "simulation_end_fnc": simulation_end_fnc,
            "initialize_timeout": initialize_timeout,
            "close_timeout": close_timeout,
            "session_end_timeout": session_end_timeout,
            "preload_modules": tuple(preload_modules),
            "ping_interval": 2.5,
            "ping_timeout": 60,
            "high_ping_threshold": 0.5,
            "memory_warn_mb": memory_warn_mb,
            "memory_limit_mb": memory_limit_mb,
            "http_proxy": http_proxy,
            "env": env,
            "mp_ctx": mp_ctx,
            "loop": loop,
        }
        self._pool = _core.ProcPool(
            self,
            num_idle_processes=num_idle_processes,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
        )

    @property
    def processes(self) -> list[JobExecutor]:
        return self._pool.processes

    @property
    def _jobs_waiting_for_process(self) -> int:
        return self._pool.jobs_waiting

    def get_by_job_id(self, job_id: str) -> JobExecutor | None:
        return next(
            (x for x in self.processes if x.running_job and x.running_job.job.id == job_id),
            None,
        )

    async def start(self) -> None:
        await self._pool.start()

    async def aclose(self) -> None:
        await self._pool.aclose()

    async def join_jobs(self) -> None:
        """Waits until every process a job was launched on exited (the drain)."""
        await self._pool.join_jobs()

    async def launch_job(self, info: RunningJobInfo) -> None:
        await self._pool.launch_job(info.job.id, info)

    def set_target_idle_processes(self, num_idle_processes: int) -> None:
        self._pool.target_idle_processes = num_idle_processes

    @property
    def target_idle_processes(self) -> int:
        return self._pool.target_idle_processes

    def _core_create(self) -> JobExecutor:
        return JobExecutor(**self._executor_args)


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
    "JobExecutor",
]
