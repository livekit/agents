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
from . import channel, job_proc_lazy_main, proto
from .inference_executor import InferenceExecutor
from .job_executor import JobStatus


@dataclass
class _ProcOpts:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]]
    initialize_timeout: float
    close_timeout: float
    ping_interval: float
    high_ping_threshold: float


class ThreadJobExecutor:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        inference_executor: InferenceExecutor | None,
        initialize_timeout: float,
        close_timeout: float,
        ping_interval: float,
        high_ping_threshold: float,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._opts = _ProcOpts(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            ping_interval=ping_interval,
            high_ping_threshold=high_ping_threshold,
        )

        self._user_args: Any | None = None
        self._job_status: JobStatus | None = None
        self._running_job: RunningJobInfo | None = None

        self._main_atask: asyncio.Task[None] | None = None
        self._initialize_fut = asyncio.Future[None]()
        self._closing = False
        self._lock = asyncio.Lock()

        self._inference_executor = inference_executor
        self._inference_tasks: list[asyncio.Task[None]] = []

    @property
    def status(self) -> JobStatus:
        if self._job_status is None:
            raise RuntimeError("job status not available")

        return self._job_status

    @property
    def started(self) -> bool:
        return self._main_atask is not None

    @property
    def user_arguments(self) -> Any | None:
        return self._user_args

    @user_arguments.setter
    def user_arguments(self, value: Any | None) -> None:
        self._user_args = value

    @property
    def running_job(self) -> RunningJobInfo | None:
        return self._running_job

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

            targs = job_proc_lazy_main.ThreadStartArgs(
                mp_cch=mp_cch,
                initialize_process_fnc=self._opts.initialize_process_fnc,
                job_entrypoint_fnc=self._opts.job_entrypoint_fnc,
                user_arguments=self._user_args,
                join_fnc=_on_join,
            )

            self._thread = t = threading.Thread(
                target=job_proc_lazy_main.thread_main,
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
            logger.error(
                "job shutdown is taking too much time..", extra=self.logging_extra()
            )

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def _do_inference_task(self, inf_req: proto.InferenceRequest) -> None:
        if self._inference_executor is None:
            logger.warning("inference request received but no inference executor")
            await channel.asend_message(
                self._pch,
                proto.InferenceResponse(
                    request_id=inf_req.request_id, error="no inference executor"
                ),
            )
            return

        try:
            inf_res = await self._inference_executor.do_inference(
                inf_req.method, inf_req.data
            )
            await channel.asend_message(
                self._pch,
                proto.InferenceResponse(request_id=inf_req.request_id, data=inf_res),
            )
        except Exception as e:
            await channel.asend_message(
                self._pch,
                proto.InferenceResponse(request_id=inf_req.request_id, error=str(e)),
            )

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the executor"""
        if self._running_job is not None:
            raise RuntimeError("executor already has a running job")

        if not self._initialize_fut.done():
            raise RuntimeError("executor not initialized")

        self._running_job = info
        self._job_status = JobStatus.RUNNING

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

        ping_task = asyncio.create_task(self._ping_task())
        monitor_task = asyncio.create_task(self._monitor_task())

        await self._join_fut
        await utils.aio.gracefully_cancel(ping_task, monitor_task)
        await utils.aio.gracefully_cancel(*self._inference_tasks)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

        self._job_status = JobStatus.SUCCESS

    @utils.log_exceptions(logger=logger)
    async def _monitor_task(self) -> None:
        while True:
            try:
                msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.DuplexClosed:
                break

            if isinstance(msg, proto.PongResponse):
                delay = utils.time_ms() - msg.timestamp
                if delay > self._opts.high_ping_threshold * 1000:
                    logger.warning(
                        "job executor is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

            if isinstance(msg, proto.Exiting):
                logger.debug(
                    "job exiting", extra={"reason": msg.reason, **self.logging_extra()}
                )

            if isinstance(msg, proto.InferenceRequest):
                self._inference_tasks.append(
                    asyncio.create_task(self._do_inference_task(msg))
                )

    @utils.log_exceptions(logger=logger)
    async def _ping_task(self) -> None:
        ping_interval = utils.aio.interval(self._opts.ping_interval)
        while True:
            await ping_interval.tick()
            try:
                await channel.asend_message(
                    self._pch, proto.PingRequest(timestamp=utils.time_ms())
                )
            except utils.aio.duplex_unix.DuplexClosed:
                break

    def logging_extra(self):
        extra: dict[str, Any] = {
            "tid": self._thread.native_id,
        }
        if self._running_job:
            extra["job_id"] = self._running_job.job.id

        return extra
