"""The worker's job processes, as the core (``_core.ProcSupervisor``,
``_core.ProcPool``) supervises and pools them: this module spawns them (``multiprocessing``, the
forkserver's preloads), samples their memory (psutil) and hands the core the user's callbacks.
What runs inside a process is ``ipc.child``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import signal
import socket
import threading
import time
from collections.abc import Awaitable, Callable, Generator
from enum import Enum
from multiprocessing.context import BaseContext
from types import FrameType
from typing import Any, Literal

import psutil

from .. import _core, utils
from ..job import JobContext, JobExecutorType, RunningJobInfo
from ..log import logger
from ..telemetry import metrics
from ..utils import shortuuid
from . import child

__all__ = ["JobExecutor", "JobStatus", "ProcPool"]


class JobStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


_mask_ctrl_c_refcount = 0
_mask_ctrl_c_original: Callable[[int, FrameType | None], Any] | int | None = signal.SIG_DFL


@contextlib.contextmanager
def _mask_ctrl_c() -> Generator[None, None, None]:
    """Temporarily ignore SIGINT so forked/spawned children inherit SIG_IGN.

    Unlike pthread_sigmask (per-thread), signal.signal is process-wide and SIG_IGN is preserved
    across exec() per POSIX, so children start with SIGINT ignored whichever thread forks. It is
    refcounted so concurrent starts (the pool warming several processes) don't clobber each
    other's saved handler, and only touched from the main thread (signal.signal's rule)."""
    global _mask_ctrl_c_refcount, _mask_ctrl_c_original

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    if _mask_ctrl_c_refcount == 0:
        _mask_ctrl_c_original = signal.signal(signal.SIGINT, signal.SIG_IGN)
    _mask_ctrl_c_refcount += 1
    try:
        yield
    finally:
        _mask_ctrl_c_refcount -= 1
        if _mask_ctrl_c_refcount == 0:
            signal.signal(signal.SIGINT, _mask_ctrl_c_original)


class _SupervisedProc:
    """A spawned process (or job thread) the core supervises (``_core.ProcSupervisor``): its
    initialization, pings, memory, messages and close, until it exited. The ``_core_*`` methods
    are the core's calls."""

    _pch: _core.ProcSupervisor
    _process_kind = "job"
    # a process; a job thread can't be killed, nor is it in the metrics
    _killable = True

    def __init__(
        self,
        *,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
        http_proxy: str | None,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._mp_ctx = mp_ctx
        # the core's supervision of the process
        self._opts = {
            "initialize_timeout": initialize_timeout,
            "close_timeout": close_timeout,
            "memory_warn_mb": memory_warn_mb,
            "memory_limit_mb": memory_limit_mb,
            "ping_interval": ping_interval,
            "ping_timeout": ping_timeout,
            "high_ping_threshold": high_ping_threshold,
        }
        self._http_proxy = http_proxy
        self._exitcode: int | None = None
        self._pid: int | None = None
        self._spawn_time: float | None = None
        self._psutil_process: psutil.Process | None = None
        self._start_atask: asyncio.Task[None] | None = None
        # the core's run of the process, from its initialization to its exit
        self._supervise_atask: asyncio.Future[None] | None = None
        self._closing = False

    def _create_process(self, cch: socket.socket) -> Any:
        raise NotImplementedError

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
    def uptime(self) -> float:
        """seconds since spawn, 0.0 if not started yet."""
        return 0.0 if self._spawn_time is None else time.monotonic() - self._spawn_time

    async def start(self) -> None:
        """start the supervised process"""
        if self._start_atask is not None:
            raise RuntimeError("process already started")
        if self._closing:
            raise RuntimeError("process is closed")
        # keep a handle so aclose() can await an abandoned (cancelled) start()
        self._start_atask = asyncio.create_task(self._start())
        await asyncio.shield(self._start_atask)

    async def _start(self) -> None:
        # the control socket: the child's end goes to the process, the parent's to the core
        pch, cch = _core.ipc_pair()
        child_end = socket.socket(fileno=cch)
        try:
            self._proc = self._create_process(child_end)
            # SIG_IGN set process-wide before the fork is inherited (and kept across exec), so
            # no KeyboardInterrupt lands in the child before it installs its own handlers
            with _mask_ctrl_c():
                await self._loop.run_in_executor(None, self._proc.start)
        except BaseException:
            socket.socket(fileno=pch).close()
            raise
        finally:
            child_end.close()

        self._pid = self._proc.pid
        self._spawn_time = time.monotonic()
        self._pch = _core.ProcSupervisor(
            pch,
            self,
            pid=self._pid or 0,
            process_kind=self._process_kind,
            killable=self._killable,
            **self._opts,
        )
        joined = asyncio.Future[None]()

        def _sync_run() -> None:
            self._proc.join()
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(joined.set_result, None)

        threading.Thread(target=_sync_run, name="proc_join_thread").start()
        self._supervise_atask = self._pch.run(joined, self)

    async def join(self) -> None:
        """wait for the process to finish"""
        if self._supervise_atask is None:
            raise RuntimeError("process not started")
        await asyncio.shield(self._supervise_atask)

    async def initialize(self) -> None:
        """initialize the process: the core sends it an InitializeRequest and waits for its
        answer, killing it when late (``asyncio.TimeoutError``)"""
        elapsed_time = await self._pch.initialize(
            asyncio_debug=self._loop.get_debug(), http_proxy=self._http_proxy
        )
        if self._killable:
            metrics.proc_initialized(time_elapsed=elapsed_time)

    async def aclose(self) -> None:
        """attempt to gracefully close the supervised process: an initialized one is asked to
        shut down (and killed when late), any other is killed"""
        if self._start_atask is not None and not self._start_atask.done():
            # start() may have been cancelled mid-shield; wait so the started check can't race
            await asyncio.wait([self._start_atask])
        if self._supervise_atask is None:
            return
        self._closing = True
        await self._pch.aclose()

    async def kill(self) -> None:
        """forcefully kill the supervised process"""
        if self._supervise_atask is None:
            raise RuntimeError("process not started")
        self._closing = True
        await self._pch.kill()

    def is_alive(self) -> bool:
        return self._core_is_alive()

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None:
        logger.log(level, msg, extra={**extra, **self.logging_extra()}, exc_info=exc)

    def _core_is_alive(self) -> bool:
        try:
            return bool(self._proc.is_alive())
        except ValueError:
            # the process object is closed once it was reaped
            return False

    def _core_kill(self) -> None:
        with contextlib.suppress(ValueError):
            # a terminate on Windows
            self._proc.kill()

    def _core_has_running_job(self) -> bool:
        return False

    def _core_reap(self) -> int | None:
        self._exitcode = self._proc.exitcode
        self._proc.close()
        return self._exitcode

    def _sample_memory_mb(self) -> tuple[float, str]:
        """The process's real footprint, preferring the metrics that don't double-count shared
        pages: PSS (Linux) divides the pages shared with the forkserver among their sharers, USS
        (elsewhere) counts the private ones, RSS (all of them) is the last resort. Blocking (the
        kernel walks the process's page tables): the core calls it off its runtime every 5 s."""
        process = self._psutil_process
        if process is None:
            process = self._psutil_process = psutil.Process(self._pid)
        try:
            memory_info: Any = process.memory_full_info()
        except (psutil.AccessDenied, NotImplementedError):
            memory_info = process.memory_info()
        for metric in "pss", "uss", "rss":
            if memory_bytes := getattr(memory_info, metric, 0):
                return memory_bytes / (1024 * 1024), metric
        return 0, "unknown"

    def logging_extra(self) -> dict[str, Any]:
        return {"pid": self.pid}


class _JobThread:
    """A job thread as the supervision drives a process: started, joined, never killed."""

    pid = None
    exitcode = 0

    def __init__(self, args: child.JobArgs) -> None:
        self._thread = threading.Thread(
            target=child.thread_main, args=(args,), name="job_thread_runner"
        )
        self.native_id: int | None = None

    def start(self) -> None:
        self._thread.start()
        self.native_id = self._thread.native_id

    def join(self) -> None:
        self._thread.join()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


class JobExecutor(_SupervisedProc):
    """One job's process, or (``thread``) a thread of this process: the core supervises it as a
    process except that a thread cannot be killed (a late initialization, ack or exit is
    reported and waited for), has no pong deadline nor memory limit, and is not in the
    metrics."""

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
        memory_warn_mb: float = 0,
        memory_limit_mb: float = 0,
        ping_interval: float,
        ping_timeout: float = 0,
        high_ping_threshold: float,
        http_proxy: str | None,
        mp_ctx: BaseContext | None = None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            memory_warn_mb=0 if thread else memory_warn_mb,
            memory_limit_mb=0 if thread else memory_limit_mb,
            ping_interval=ping_interval,
            # no pong deadline: a thread cannot be killed for being late
            ping_timeout=0 if thread else ping_timeout,
            high_ping_threshold=high_ping_threshold,
            http_proxy=http_proxy,
            mp_ctx=mp_ctx or mp.get_context(),
            loop=loop,
        )
        self._thread = thread
        self._killable = not thread
        self.user_arguments: Any | None = None
        self._job_status: JobStatus | None = None
        self._running_job: RunningJobInfo | None = None
        self._args = child.JobArgs(
            job_entrypoint_fnc=job_entrypoint_fnc,
            session_end_fnc=session_end_fnc,
            simulation_end_fnc=simulation_end_fnc,
            session_end_timeout=session_end_timeout,
        )
        self._id = shortuuid("THEXEC_" if thread else "PCEXEC_")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> JobStatus:
        if self._job_status is None:
            raise RuntimeError("job status not available")
        return self._job_status

    @property
    def running_job(self) -> RunningJobInfo | None:
        return self._running_job

    def _create_process(self, cch: socket.socket) -> Any:
        from ..cli import _log_config

        self._args.user_arguments = self.user_arguments
        if self._thread:
            # the thread shares this process's logging, and keeps its own copy of the socket the
            # parent closes once started
            return _JobThread(child.JobArgs(**{**vars(self._args), "mp_cch": cch.dup()}))
        levels = {"root": logging.getLogger().level}
        for log in logging.Logger.manager.loggerDict.values():
            if isinstance(log, logging.Logger):
                levels[log.name] = log.level
        self._args.mp_cch = cch
        self._args.log_config = _log_config
        self._args.logger_levels = levels
        return self._mp_ctx.Process(target=child.job_main, args=(self._args,), name="job_proc")

    def _core_has_running_job(self) -> bool:
        return self._running_job is not None

    def _core_reap(self) -> int | None:
        exitcode = super()._core_reap()
        if self._running_job and self._killable:
            metrics.job_ended()
        self._job_status = JobStatus.SUCCESS if exitcode == 0 else JobStatus.FAILED
        return exitcode

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")
        if not self.started or not self._pch.initialized:
            raise RuntimeError("process not initialized")
        if self._killable:
            metrics.job_started()
        self._job_status = JobStatus.RUNNING
        self._running_job = info
        try:
            await self._pch.start_job(info._to_wire())
        except Exception:
            self._running_job = None
            self._job_status = None
            if self._killable:
                metrics.job_ended()
            raise

    def logging_extra(self) -> dict[str, Any]:
        if self._thread:
            proc = getattr(self, "_proc", None)
            extra: dict[str, Any] = {"tid": proc.native_id if proc else None}
        else:
            extra = super().logging_extra()
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
        job_executor_type: JobExecutorType,
        mp_ctx: BaseContext,
        memory_warn_mb: float,
        memory_limit_mb: float,
        http_proxy: str | None,
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
            "ping_interval": 2.5,
            "ping_timeout": 60,
            "high_ping_threshold": 0.5,
            "memory_warn_mb": memory_warn_mb,
            "memory_limit_mb": memory_limit_mb,
            "http_proxy": http_proxy,
            "mp_ctx": mp_ctx,
            "loop": loop,
        }
        self._pool = _core.ProcPool(
            self, num_idle_processes=num_idle_processes, initialize_timeout=initialize_timeout
        )

    @property
    def processes(self) -> list[JobExecutor]:
        return self._pool.processes  # type: ignore[no-any-return]

    @property
    def _jobs_waiting_for_process(self) -> int:
        return self._pool.jobs_waiting  # type: ignore[no-any-return]

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
        await self._pool.launch_job(info)

    def set_target_idle_processes(self, num_idle_processes: int) -> None:
        self._pool.target_idle_processes = num_idle_processes

    @property
    def target_idle_processes(self) -> int:
        return self._pool.target_idle_processes  # type: ignore[no-any-return]

    def _core_create(self) -> JobExecutor:
        return JobExecutor(**self._executor_args)

    def _core_init_failed(self, proc: JobExecutor, error: BaseException) -> None:
        logger.error("error initializing process", extra=proc.logging_extra(), exc_info=error)
