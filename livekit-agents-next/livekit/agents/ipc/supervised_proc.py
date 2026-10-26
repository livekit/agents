from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum
from multiprocessing.context import BaseContext
from types import FrameType
from typing import Any

import psutil

from .. import _core
from ..log import logger
from ..telemetry import metrics
from ..utils.aio import duplex_unix
from . import channel, proto
from .log_queue import LogQueueListener

_mask_ctrl_c_refcount = 0
_mask_ctrl_c_original: Callable[[int, FrameType | None], Any] | int | None = signal.SIG_DFL


class SupervisedProcKind(str, Enum):
    JOB = "job"
    INFERENCE = "inference"

    def __str__(self) -> str:
        return self.value


@contextlib.contextmanager
def _mask_ctrl_c() -> Generator[None, None, None]:
    """Temporarily ignore SIGINT so forked/spawned children inherit SIG_IGN.

    Unlike pthread_sigmask (per-thread), signal.signal is process-wide and
    SIG_IGN is preserved across exec() per POSIX — so children start with
    SIGINT ignored regardless of which thread performs the fork.

    Uses refcounting so concurrent async callers (e.g. proc pool warming
    multiple processes) don't clobber each other's saved handler.

    signal.signal() can only be called from the main thread.
    Keep the critical section *tiny* (just around Process.start()).
    """
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


@dataclass
class _ProcOpts:
    initialize_timeout: float
    close_timeout: float
    memory_warn_mb: float
    memory_limit_mb: float
    ping_interval: float
    ping_timeout: float
    high_ping_threshold: float
    http_proxy: str | None


class SupervisedProc(ABC):
    """A spawned process (or job thread) the core supervises (``_core.ProcSupervisor``): its
    initialization, pings, memory, messages and close, until it exited. The ``_core_*`` methods
    are the core's calls."""

    _pch: _core.ProcSupervisor
    # a process; the thread executor's runner can't be killed, nor is it in the metrics
    _killable = True
    _metered = True
    _runner = "process"

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
        self._opts = _ProcOpts(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            memory_warn_mb=memory_warn_mb,
            memory_limit_mb=memory_limit_mb,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            high_ping_threshold=high_ping_threshold,
            http_proxy=http_proxy,
        )

        self._exitcode: int | None = None
        self._pid: int | None = None
        self._spawn_time: float | None = None

        self._psutil_process: psutil.Process | None = None

        self._start_atask: asyncio.Task[None] | None = None
        # the core's run of the process, from its initialization to its exit
        self._supervise_atask: asyncio.Future[None] | None = None
        self._closing = False

    @abstractmethod
    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process: ...

    @property
    def enabled_stack_trace_dump(self) -> bool:
        return os.getenv("LK_DUMP_STACK_TRACES", "0").lower() not in ("0", "false", "no")

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
    @abstractmethod
    def process_kind(self) -> SupervisedProcKind: ...

    @property
    def uptime(self) -> float:
        """seconds since spawn, 0.0 if not started yet."""
        if self._spawn_time is None:
            return 0.0
        return time.monotonic() - self._spawn_time

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
        def _add_proc_ctx_log(record: logging.LogRecord) -> None:
            extra = self.logging_extra()
            for key, value in extra.items():
                setattr(record, key, value)

        mp_pch, mp_cch = socket.socketpair()
        mp_log_pch, mp_log_cch = socket.socketpair()

        sockets = (mp_pch, mp_cch, mp_log_pch, mp_log_cch)
        log_listener: LogQueueListener | None = None
        try:
            log_pch = duplex_unix._Duplex.open(mp_log_pch)
            log_listener = LogQueueListener(log_pch, _add_proc_ctx_log)
            log_listener.start()

            self._proc = self._create_process(mp_cch, mp_log_cch)

            # Set SIG_IGN process-wide before forking so the child inherits it
            # (SIG_IGN is preserved across exec per POSIX). This prevents
            # KeyboardInterrupt during the child's bootstrap phase before
            # it can install its own signal handlers.
            with _mask_ctrl_c():
                await self._loop.run_in_executor(None, self._proc.start)
        except Exception:
            for s in sockets:
                with contextlib.suppress(OSError):
                    s.close()

            if log_listener is not None:
                with contextlib.suppress(duplex_unix.DuplexClosed):
                    log_listener.stop()
            raise

        mp_log_cch.close()
        mp_cch.close()

        self._pid = self._proc.pid
        self._spawn_time = time.monotonic()
        self._pch = _core.ProcSupervisor(
            mp_pch.detach(),
            self,
            pid=self._pid or 0,
            process_kind=str(self.process_kind),
            initialize_timeout=self._opts.initialize_timeout,
            close_timeout=self._opts.close_timeout,
            memory_warn_mb=self._opts.memory_warn_mb,
            memory_limit_mb=self._opts.memory_limit_mb,
            ping_interval=self._opts.ping_interval,
            ping_timeout=self._opts.ping_timeout,
            high_ping_threshold=self._opts.high_ping_threshold,
            killable=self._killable,
        )
        joined = asyncio.Future[None]()

        def _sync_run() -> None:
            self._proc.join()
            log_listener.stop()
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
        """initialize the process, this is sending a InitializeRequest message and waiting for a
        InitializeResponse with a timeout"""
        await channel.asend_message(
            self._pch,
            proto.InitializeRequest(
                asyncio_debug=self._loop.get_debug(),
                ping_interval=self._opts.ping_interval,
                ping_timeout=self._opts.ping_timeout,
                high_ping_threshold=self._opts.high_ping_threshold,
                http_proxy=self._opts.http_proxy or "",
            ),
        )

        # should be channel.ChannelClosed most of the time when it raises (or init_res error)
        elapsed_time = await self._pch.initialize()
        if elapsed_time is None:
            # the core killed the process
            raise asyncio.TimeoutError(f"{self._runner} initialization timed out")
        if self._metered:
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

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None:
        logger.log(level, msg, extra={**extra, **self.logging_extra()}, exc_info=exc)

    def _core_is_alive(self) -> bool:
        try:
            return self._proc.is_alive()
        except ValueError:
            return False

    def _core_kill(self) -> None:
        with contextlib.suppress(ValueError):
            # a terminate on Windows
            self._proc.kill()

    def _core_has_running_job(self) -> bool:
        # _running_job only exists on the job executor subclass, not the inference one
        return getattr(self, "_running_job", None) is not None

    def _core_reap(self) -> int | None:
        self._exitcode = self._proc.exitcode
        self._proc.close()
        return self._exitcode

    async def _core_inference(self, method: str, data: bytes) -> bytes | None:
        raise RuntimeError("no inference executor")

    def _sample_memory_mb(self) -> tuple[float, str]:
        """Real footprint of the process, preferring metrics that don't
        double-count shared pages. RSS counts copy-on-write pages inherited
        from the forkserver and file-backed library pages at memory_info weight, so it
        overstates what the process actually consumes and oversums across
        processes. PSS (proportional, Linux only) divides shared pages among
        their sharers and sums correctly to physical use; USS (private only) is
        the cross-platform fallback since psutil exposes no PSS on macOS/Windows.
        RSS is a last resort if the extended metrics are unavailable.

        Synchronous and blocking: the PSS/USS path walks the target process's
        page tables in the kernel, which scales with its mapping count and is
        ~ms for a large child. The core calls it off its runtime every 5 s."""
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
        extra: dict[str, Any] = {
            "pid": self.pid,
        }

        return extra
