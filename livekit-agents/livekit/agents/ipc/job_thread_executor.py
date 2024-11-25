from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .. import utils
from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from ..utils.aio import duplex_unix
from . import channel, job_main, proto
from .job_executor import (
    JobExecutorError_ShutdownTimeout,
    JobExecutorError_Unresponsive,
    RunStatus,
)


@dataclass
class _ProcOpts:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]]
    initialize_timeout: float
    close_timeout: float


class ThreadJobExecutor:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        initialize_timeout: float,
        close_timeout: float,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._opts = _ProcOpts(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
        )

        self._user_args: Any | None = None
        self._running_job: RunningJobInfo | None = None
        self._exception: Exception | None = None

        self._main_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._initialize_fut = asyncio.Future[None]()

        self._lock = asyncio.Lock()

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
        if self.started:
            raise RuntimeError("runner already started")

        if self._closing:
            raise RuntimeError("runner is closed")

        await asyncio.shield(self._start())

    async def _start(self) -> None:
        async with self._lock:
            # to simplify the runners implementation, we also use a duplex in the threaded executor
            # (ThreadedRunners), so we can use the same protocol
            mp_pch, mp_cch = socket.socketpair()
            self._pch = await duplex_unix._AsyncDuplex.open(mp_pch)

            self._join_fut = asyncio.Future[None]()

            def _on_join() -> None:
                with contextlib.suppress(RuntimeError):
                    self._loop.call_soon_threadsafe(self._join_fut.set_result, None)

            targs = job_main.ThreadStartArgs(
                mp_cch=mp_cch,
                initialize_process_fnc=self._opts.initialize_process_fnc,
                job_entrypoint_fnc=self._opts.job_entrypoint_fnc,
                user_arguments=self._user_args,
                asyncio_debug=self._loop.get_debug(),
                join_fnc=_on_join,
            )

            self._thread = t = threading.Thread(
                target=job_main.thread_main,
                args=(targs,),
                name="job_thread_runner",
            )
            t.start()

            self._main_atask = asyncio.create_task(self._main_task())

    async def join(self) -> None:
        """wait for the thread to finish"""
        if not self.started:
            raise RuntimeError("runner not started")

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def initialize(self) -> None:
        await channel.asend_message(self._pch, proto.InitializeRequest())

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
                asyncio.TimeoutError("runner initialization timed out")
            )
            logger.error(
                "job initialization is taking too much time..",
                extra=self.logging_extra(),
            )
            raise
        except Exception as e:  # should be channel.ChannelClosed most of the time
            self._initialize_fut.set_exception(e)
            raise
        else:
            self._initialize_fut.set_result(None)

    async def aclose(self) -> None:
        """
        attempt to gracefully close the job. warn if it takes too long to close
        (in the threaded executor, the job can't be "killed")
        """
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
            self._exception = JobExecutorError_ShutdownTimeout()
            logger.error(
                "job shutdown is taking too much time..", extra=self.logging_extra()
            )

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the executor"""
        if self._running_job is not None:
            raise RuntimeError("executor already has a running job")

        self._running_job = info
        start_req = proto.StartJobRequest()
        start_req.running_job = info
        await channel.asend_message(self._pch, start_req)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            await self._initialize_fut
        except asyncio.TimeoutError:
            pass  # this happens when the initialization takes longer than self._initialize_timeout
        except Exception:
            pass  # initialization failed

        pong_timeout = utils.aio.sleep(proto.PING_TIMEOUT)
        ping_task = asyncio.create_task(self._ping_pong_task(pong_timeout))
        monitor_task = asyncio.create_task(self._monitor_task(pong_timeout))

        await self._join_fut
        await utils.aio.gracefully_cancel(ping_task, monitor_task)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

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
                        "job executor is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

                with contextlib.suppress(utils.aio.SleepFinished):
                    pong_timeout.reset()

            if isinstance(msg, proto.Exiting):
                logger.debug(
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
            self._exception = JobExecutorError_Unresponsive()
            logger.error("job is unresponsive..", extra=self.logging_extra())

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
            "tid": self._thread.native_id,
        }
        if self._running_job:
            extra["job_id"] = self._running_job.job.id

        return extra
