from __future__ import annotations

import asyncio
import contextlib
import logging
import pickle
import socket
import sys
import threading
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Awaitable, Callable

from .. import utils
from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from ..utils.aio import duplex_unix
from . import channel, job_main, proc_lazy_main, proto
from .job_executor import (
    JobExecutorError_Runtime,
    JobExecutorError_ShutdownTimeout,
    JobExecutorError_Unresponsive,
    RunStatus,
)


class LogQueueListener:
    def __init__(
        self,
        duplex: utils.aio.duplex_unix._Duplex,
        prepare_fnc: Callable[[logging.LogRecord], None],
    ):
        self._thread: threading.Thread | None = None
        self._duplex = duplex
        self._prepare_fnc = prepare_fnc

    def start(self) -> None:
        self._thread = threading.Thread(target=self._monitor, name="ipc_log_listener")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return

        self._duplex.close()
        self._thread.join()
        self._thread = None

    def handle(self, record: logging.LogRecord) -> None:
        self._prepare_fnc(record)

        lger = logging.getLogger(record.name)
        if not lger.isEnabledFor(record.levelno):
            return

        lger.callHandlers(record)

    def _monitor(self):
        while True:
            try:
                data = self._duplex.recv_bytes()
            except utils.aio.duplex_unix.DuplexClosed:
                break

            record = pickle.loads(data)
            self.handle(record)


@dataclass
class _ProcOpts:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]]
    mp_ctx: BaseContext
    initialize_timeout: float
    close_timeout: float


class ProcJobExecutor:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        initialize_timeout: float,
        close_timeout: float,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._opts = _ProcOpts(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            mp_ctx=mp_ctx,
        )

        self._user_args: Any | None = None
        self._running_job: RunningJobInfo | None = None
        self._exitcode: int | None = None
        self._pid: int | None = None
        self._exception: Exception | None = None

        self._main_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._kill_sent = False
        self._initialize_fut = asyncio.Future[None]()

        self._lock = asyncio.Lock()

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
        return self._main_atask is not None

    @property
    def start_arguments(self) -> Any | None:
        return self._user_args

    @start_arguments.setter
    def start_arguments(self, value: Any | None) -> None:
        self._user_args = value

    @property
    def running_job(self) -> RunningJobInfo | None:
        return self._running_job

    @property
    def exception(self) -> Exception | None:
        return self._exception

    @property
    def run_status(self) -> RunStatus:
        if not self._running_job:
            if self.started:
                return RunStatus.WAITING_FOR_JOB
            else:
                return RunStatus.STARTING

        if not self._main_atask:
            return RunStatus.STARTING

        if self._main_atask.done():
            if self.exception:
                return RunStatus.FINISHED_FAILED
            else:
                return RunStatus.FINISHED_CLEAN
        else:
            return RunStatus.RUNNING_JOB

    async def start(self) -> None:
        """start the job process"""
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

            self._proc_args = job_main.ProcStartArgs(
                initialize_process_fnc=self._opts.initialize_process_fnc,
                job_entrypoint_fnc=self._opts.job_entrypoint_fnc,
                log_cch=mp_log_cch,
                mp_cch=mp_cch,
                asyncio_debug=self._loop.get_debug(),
                user_arguments=self._user_args,
            )

            self._proc = self._opts.mp_ctx.Process(  # type: ignore
                target=proc_lazy_main.proc_main,
                args=(self._proc_args,),
                name="job_proc",
            )

            self._proc.start()
            mp_log_cch.close()
            mp_cch.close()

            self._pid = self._proc.pid
            self._join_fut = asyncio.Future[None]()

            def _sync_run():
                self._proc.join()
                log_listener.stop()
                try:
                    self._loop.call_soon_threadsafe(self._join_fut.set_result, None)
                except RuntimeError:
                    pass

            thread = threading.Thread(target=_sync_run, name="proc_join_thread")
            thread.start()
            self._main_atask = asyncio.create_task(self._main_task())

    async def join(self) -> None:
        """wait for the job process to finish"""
        if not self.started:
            raise RuntimeError("process not started")

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def initialize(self) -> None:
        """initialize the job process, this is calling the user provided initialize_process_fnc
        raise asyncio.TimeoutError if initialization times out"""
        await channel.asend_message(self._pch, proto.InitializeRequest())

        # wait for the process to become ready
        try:
            init_res = await asyncio.wait_for(
                channel.arecv_message(self._pch, proto.IPC_MESSAGES),
                timeout=self._opts.initialize_timeout,
            )
            assert isinstance(
                init_res, proto.InitializeResponse
            ), "first message must be InitializeResponse"
        except asyncio.TimeoutError:
            self._initialize_fut.set_exception(
                asyncio.TimeoutError("process initialization timed out")
            )
            logger.error(
                "initialization timed out, killing job", extra=self.logging_extra()
            )
            self._send_kill_signal()
            raise
        except Exception as e:  # should be channel.ChannelClosed most of the time
            self._exception = JobExecutorError_Runtime()
            self._initialize_fut.set_exception(e)
            raise
        else:
            self._initialize_fut.set_result(None)

    async def aclose(self) -> None:
        """attempt to gracefully close the job process"""
        if not self.started:
            return

        self._closing = True
        with contextlib.suppress(utils.aio.duplex_unix.DuplexClosed):
            await channel.asend_message(self._pch, proto.ShutdownRequest())

        try:
            if self._main_atask:
                await asyncio.wait_for(
                    asyncio.shield(self._main_atask), timeout=self._opts.close_timeout
                )
        except asyncio.TimeoutError:
            logger.error(
                "process did not exit in time, killing job", extra=self.logging_extra()
            )
            self._exception = JobExecutorError_ShutdownTimeout()
            self._send_kill_signal()

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def kill(self) -> None:
        """forcefully kill the job process"""
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        self._send_kill_signal()

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")

        self._running_job = info
        start_req = proto.StartJobRequest()
        start_req.running_job = info
        await channel.asend_message(self._pch, start_req)

    def _send_kill_signal(self) -> None:
        """forcefully kill the job process"""
        try:
            if not self._proc.is_alive():
                return
        except ValueError:
            return

        logger.info("killing job process", extra=self.logging_extra())
        if sys.platform == "win32":
            self._proc.terminate()
        else:
            self._proc.kill()

        self._kill_sent = True

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            await self._initialize_fut
        except asyncio.TimeoutError:
            pass  # this happens when the initialization takes longer than self._initialize_timeout
        except Exception:
            pass  # initialization failed

        # the process is killed if it doesn't respond to ping requests
        pong_timeout = utils.aio.sleep(proto.PING_TIMEOUT)
        ping_task = asyncio.create_task(self._ping_pong_task(pong_timeout))
        monitor_task = asyncio.create_task(self._monitor_task(pong_timeout))

        await self._join_fut
        self._exitcode = self._proc.exitcode
        self._proc.close()
        await utils.aio.gracefully_cancel(ping_task, monitor_task)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

        if self._exitcode != 0 and not self._kill_sent:
            self._exception = JobExecutorError_Runtime()
            logger.error(
                f"job process exited with non-zero exit code {self.exitcode}",
                extra=self.logging_extra(),
            )

    @utils.log_exceptions(logger=logger)
    async def _monitor_task(self, pong_timeout: utils.aio.Sleep) -> None:
        while True:
            try:
                msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.DuplexClosed:
                break

            if isinstance(msg, proto.PongResponse):
                delay = utils.time_ms() - msg.timestamp
                if delay > proto.HIGH_PING_THRESHOLD * 1000:
                    logger.warning(
                        "job process is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

                with contextlib.suppress(utils.aio.SleepFinished):
                    pong_timeout.reset()

            if isinstance(msg, proto.Exiting):
                logger.info(
                    "job exiting", extra={"reason": msg.reason, **self.logging_extra()}
                )

    @utils.log_exceptions(logger=logger)
    async def _ping_pong_task(self, pong_timeout: utils.aio.Sleep) -> None:
        ping_interval = utils.aio.interval(proto.PING_INTERVAL)

        async def _send_ping_co():
            while True:
                await ping_interval.tick()
                try:
                    await channel.asend_message(
                        self._pch, proto.PingRequest(timestamp=utils.time_ms())
                    )
                except utils.aio.duplex_unix.DuplexClosed:
                    break

        async def _pong_timeout_co():
            await pong_timeout
            logger.error("job is unresponsive, killing job", extra=self.logging_extra())
            self._exception = JobExecutorError_Unresponsive()
            self._send_kill_signal()

        tasks = [
            asyncio.create_task(_send_ping_co()),
            asyncio.create_task(_pong_timeout_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def logging_extra(self):
        extra: dict[str, Any] = {
            "pid": self.pid,
        }
        if self._running_job:
            extra["job_id"] = self._running_job.job.id

        return extra
