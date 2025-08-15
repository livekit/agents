from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any

import psutil

from ..log import logger
from ..telemetry import metrics
from ..utils import aio, log_exceptions, time_ms
from ..utils.aio import duplex_unix
from . import channel, proto
from .log_queue import LogQueueListener


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

        self._supervise_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._kill_sent = False
        self._initialize_fut = asyncio.Future[None]()
        self._lock = asyncio.Lock()

    @abstractmethod
    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process: ...

    @abstractmethod
    async def _main_task(self, ipc_ch: aio.ChanReceiver[channel.Message]) -> None: ...

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    @property
    def killed(self) -> bool:
        return self._kill_sent

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def started(self) -> bool:
        return self._supervise_atask is not None

    async def start(self) -> None:
        """start the supervised process"""
        if self.started:
            raise RuntimeError("process already started")

        if self._closing:
            raise RuntimeError("process is closed")

        await asyncio.shield(self._start())

    async def _start(self) -> None:
        def _add_proc_ctx_log(record: logging.LogRecord) -> None:
            extra = self.logging_extra()
            for key, value in extra.items():
                setattr(record, key, value)

        async with self._lock:
            mp_pch, mp_cch = socket.socketpair()
            mp_log_pch, mp_log_cch = socket.socketpair()

            self._pch = await duplex_unix._AsyncDuplex.open(mp_pch)

            log_pch = duplex_unix._Duplex.open(mp_log_pch)
            log_listener = LogQueueListener(log_pch, _add_proc_ctx_log)
            log_listener.start()

            self._proc = self._create_process(mp_cch, mp_log_cch)
            await self._loop.run_in_executor(None, self._proc.start)
            mp_log_cch.close()
            mp_cch.close()

            self._pid = self._proc.pid
            self._join_fut = asyncio.Future[None]()

            def _sync_run() -> None:
                self._proc.join()
                log_listener.stop()
                try:
                    self._loop.call_soon_threadsafe(self._join_fut.set_result, None)
                except RuntimeError:
                    pass

            thread = threading.Thread(target=_sync_run, name="proc_join_thread")
            thread.start()
            self._supervise_atask = asyncio.create_task(self._supervise_task())

    async def join(self) -> None:
        """wait for the process to finish"""
        if not self.started:
            raise RuntimeError("process not started")

        if self._supervise_atask:
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

        # wait for the process to become ready
        try:
            logger.info("initializing process", extra=self.logging_extra())
            start_time = time.perf_counter()
            init_res = await asyncio.wait_for(
                channel.arecv_message(self._pch, proto.IPC_MESSAGES),
                timeout=self._opts.initialize_timeout,
            )
            assert isinstance(init_res, proto.InitializeResponse), (
                "first message must be InitializeResponse"
            )

            if init_res.error:
                raise RuntimeError(f"process initialization failed: {init_res.error}")
            else:
                self._initialize_fut.set_result(None)

            elapsed_time = time.perf_counter() - start_time
            metrics.proc_initialized(time_elapsed=elapsed_time)
            logger.info(
                "process initialized",
                extra={
                    **self.logging_extra(),
                    "elapsed_time": round(elapsed_time, 2),
                },
            )
        except asyncio.TimeoutError:
            self._initialize_fut.set_exception(
                asyncio.TimeoutError("process initialization timed out")
            )
            self._send_kill_signal()
            raise
        except Exception as e:
            # should be channel.ChannelClosed most of the time (or init_res error)
            self._initialize_fut.set_exception(e)
            raise

    async def aclose(self) -> None:
        """attempt to gracefully close the supervised process"""
        if not self.started:
            return

        self._closing = True
        with contextlib.suppress(duplex_unix.DuplexClosed):
            await channel.asend_message(self._pch, proto.ShutdownRequest())

        try:
            if self._supervise_atask:
                await asyncio.wait_for(
                    asyncio.shield(self._supervise_atask),
                    timeout=self._opts.close_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(
                "process did not exit in time, killing process",
                extra=self.logging_extra(),
            )
            self._send_kill_signal()

        async with self._lock:
            if self._supervise_atask:
                await asyncio.shield(self._supervise_atask)

    async def kill(self) -> None:
        """forcefully kill the supervised process"""
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        self._send_kill_signal()

        async with self._lock:
            if self._supervise_atask:
                await asyncio.shield(self._supervise_atask)

    def _send_kill_signal(self) -> None:
        """forcefully kill the process"""
        try:
            if not self._proc.is_alive():
                return
        except ValueError:
            return

        logger.info("killing process", extra=self.logging_extra())
        if sys.platform == "win32":
            self._proc.terminate()
        else:
            self._proc.kill()

        self._kill_sent = True

    @log_exceptions(logger=logger)
    async def _supervise_task(self) -> None:
        try:
            await self._initialize_fut
        except asyncio.TimeoutError:
            pass  # this happens when the initialization takes longer than self._initialize_timeout
        except Exception:
            pass  # initialization failed

        # the process is killed if it doesn't respond to ping requests
        pong_timeout = aio.sleep(self._opts.ping_timeout)

        ipc_ch = aio.Chan[channel.Message]()

        main_task = asyncio.create_task(self._main_task(ipc_ch))
        read_ipc_task = asyncio.create_task(self._read_ipc_task(ipc_ch, pong_timeout))
        ping_task = asyncio.create_task(self._ping_pong_task(pong_timeout))
        read_ipc_task.add_done_callback(lambda _: ipc_ch.close())

        memory_monitor_task: asyncio.Task[None] | None = None
        if self._opts.memory_limit_mb > 0 or self._opts.memory_warn_mb > 0:
            memory_monitor_task = asyncio.create_task(self._memory_monitor_task())

        await self._join_fut
        self._exitcode = self._proc.exitcode
        self._proc.close()
        await aio.cancel_and_wait(ping_task, read_ipc_task, main_task)

        if memory_monitor_task is not None:
            await aio.cancel_and_wait(memory_monitor_task)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

        if self._exitcode != 0 and not self._kill_sent:
            logger.error(
                f"process exited with non-zero exit code {self.exitcode}",
                extra=self.logging_extra(),
            )

    @log_exceptions(logger=logger)
    async def _read_ipc_task(
        self, ipc_ch: aio.Chan[channel.Message], pong_timeout: aio.Sleep
    ) -> None:
        while True:
            try:
                msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            except duplex_unix.DuplexClosed:
                break

            if isinstance(msg, proto.PongResponse):
                delay = time_ms() - msg.timestamp
                if delay > self._opts.high_ping_threshold * 1000:
                    logger.warning(
                        "process is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

                with contextlib.suppress(aio.SleepFinished):
                    pong_timeout.reset()

            if isinstance(msg, proto.Exiting):
                logger.info(
                    "process exiting",
                    extra={"reason": msg.reason, **self.logging_extra()},
                )

            ipc_ch.send_nowait(msg)

    @log_exceptions(logger=logger)
    async def _ping_pong_task(self, pong_timeout: aio.Sleep) -> None:
        ping_interval = aio.interval(self._opts.ping_interval)

        async def _send_ping_co() -> None:
            while True:
                await ping_interval.tick()
                try:
                    await channel.asend_message(self._pch, proto.PingRequest(timestamp=time_ms()))
                except duplex_unix.DuplexClosed:
                    break

        async def _pong_timeout_co() -> None:
            await pong_timeout
            logger.error("process is unresponsive, killing process", extra=self.logging_extra())
            self._send_kill_signal()

        tasks = [
            asyncio.create_task(_send_ping_co()),
            asyncio.create_task(_pong_timeout_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await aio.cancel_and_wait(*tasks)

    @log_exceptions(logger=logger)
    async def _memory_monitor_task(self) -> None:
        """Monitor memory usage and kill the process if it exceeds the limit."""
        while not self._closing and not self._kill_sent:
            try:
                if not self._pid:
                    await asyncio.sleep(5)
                    continue

                # get process memory info
                process = psutil.Process(self._pid)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

                if self._opts.memory_limit_mb > 0 and memory_mb > self._opts.memory_limit_mb:
                    logger.error(
                        "process exceeded memory limit, killing process",
                        extra={
                            "memory_usage_mb": memory_mb,
                            "memory_limit_mb": self._opts.memory_limit_mb,
                            **self.logging_extra(),
                        },
                    )
                    self._send_kill_signal()
                elif self._opts.memory_warn_mb > 0 and memory_mb > self._opts.memory_warn_mb:
                    logger.warning(
                        "process memory usage is high",
                        extra={
                            "memory_usage_mb": memory_mb,
                            "memory_warn_mb": self._opts.memory_warn_mb,
                            "memory_limit_mb": self._opts.memory_limit_mb,
                            **self.logging_extra(),
                        },
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                if self._closing or self._kill_sent:
                    return

                logger.warning(
                    "Failed to get memory info for process",
                    extra=self.logging_extra(),
                    exc_info=e,
                )
                # don't bother rechecking if we cannot get process info
                return
            except Exception:
                if self._closing or self._kill_sent:
                    return

                logger.exception(
                    "Error in memory monitoring task",
                    extra=self.logging_extra(),
                )

            await asyncio.sleep(5)  # check every 5 seconds

    def logging_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "pid": self.pid,
        }

        return extra
