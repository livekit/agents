from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

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
    session_end_fnc: Callable[[JobContext], Awaitable[None]] | None
    initialize_timeout: float
    close_timeout: float
    ping_interval: float
    high_ping_threshold: float
    http_proxy: str | None


class ThreadJobExecutor:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        inference_executor: InferenceExecutor | None,
        initialize_timeout: float,
        close_timeout: float,
        ping_interval: float,
        high_ping_threshold: float,
        http_proxy: str | None,
        reuse_process: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._opts = _ProcOpts(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
            session_end_fnc=session_end_fnc,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            ping_interval=ping_interval,
            high_ping_threshold=high_ping_threshold,
            http_proxy=http_proxy,
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
        self._id = utils.shortuuid("THEXEC_")
        self._jobs_completed = 0
        self._baseline_memory_mb: float | None = None
        self._current_memory_mb: float | None = None
        self._reuse_process = reuse_process
        self._job_reused_event = asyncio.Event()

    @property
    def id(self) -> str:
        return self._id

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

    @property
    def jobs_completed(self) -> int:
        return self._jobs_completed

    @property
    def baseline_memory_mb(self) -> float | None:
        return self._baseline_memory_mb

    @property
    def current_memory_mb(self) -> float | None:
        return self._current_memory_mb

    @property
    def memory_growth_mb(self) -> float:
        if self._baseline_memory_mb is None or self._current_memory_mb is None:
            return 0.0
        return max(0.0, self._current_memory_mb - self._baseline_memory_mb)

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
                session_end_fnc=self._opts.session_end_fnc,
                user_arguments=self._user_args,
                join_fnc=_on_join,
                reuse_process=self._reuse_process,
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
        await channel.asend_message(
            self._pch, proto.InitializeRequest(http_proxy=self._opts.http_proxy or "")
        )

        try:
            logger.info("initializing job runner", extra=self.logging_extra())
            start_time = time.perf_counter()
            init_res = await asyncio.wait_for(
                channel.arecv_message(self._pch, proto.IPC_MESSAGES),
                timeout=self._opts.initialize_timeout,
            )
            assert isinstance(init_res, proto.InitializeResponse), (
                "first message must be InitializeResponse"
            )
            logger.info(
                "job runner initialized",
                extra={
                    **self.logging_extra(),
                    "elapsed_time": round(time.perf_counter() - start_time, 2),
                },
            )
        except asyncio.TimeoutError:
            self._initialize_fut.set_exception(
                asyncio.TimeoutError("runner initialization timed out")
            )
            raise
        except Exception as e:  # should be channel.ChannelClosed most of the time
            self._initialize_fut.set_exception(e)
            raise
        else:
            self._initialize_fut.set_result(None)

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
            inf_res = await self._inference_executor.do_inference(inf_req.method, inf_req.data)
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

    def clear_running_job(self) -> None:
        """Clear the running job (for reuse)"""
        self._running_job = None
        self._job_status = None
        # Signal that job was reused
        self._job_reused_event.set()

    async def aclose(self) -> None:
        """
        Attempt to gracefully close the executor.
        If process reuse is enabled and job completes during shutdown, skip closing.
        """
        if not self.started:
            return

        self._closing = True
        has_running_job = self._running_job is not None

        with contextlib.suppress(utils.aio.duplex_unix.DuplexClosed):
            await channel.asend_message(self._pch, proto.ShutdownRequest())

        # Wait for either thread exit OR job reuse (if reuse enabled with active job)
        try:
            if self._main_atask:
                if self._reuse_process and has_running_job:
                    # Create a task for the reuse event
                    reuse_task = asyncio.create_task(self._job_reused_event.wait())

                    # Wait for either main task completion or job reuse event
                    done, pending = await asyncio.wait(
                        [self._main_atask, reuse_task],
                        timeout=self._opts.close_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    # Check if job was reused
                    if self._job_reused_event.is_set():
                        logger.info(
                            "thread executor completed job and will be reused, skipping shutdown",
                            extra=self.logging_extra(),
                        )
                        self._closing = False
                        self._job_reused_event.clear()  # Reset for next time
                        return

                    # If we got here via timeout, raise TimeoutError
                    if not done:
                        raise asyncio.TimeoutError()
                else:
                    # Normal close without reuse consideration
                    await asyncio.wait_for(
                        asyncio.shield(self._main_atask), timeout=self._opts.close_timeout
                    )
        except asyncio.TimeoutError:
            logger.error("job shutdown is taking too much time..", extra=self.logging_extra())

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

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
        await utils.aio.cancel_and_wait(ping_task, monitor_task)
        await utils.aio.cancel_and_wait(*self._inference_tasks)

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
                logger.debug("job exiting", extra={"reason": msg.reason, **self.logging_extra()})

            if isinstance(msg, proto.InferenceRequest):
                self._inference_tasks.append(asyncio.create_task(self._do_inference_task(msg)))

            if isinstance(msg, proto.JobCompleted):
                self._handle_job_completed(msg)

    def _handle_job_completed(self, msg: proto.JobCompleted) -> None:
        """Handle job completion notification from subprocess"""
        self._jobs_completed += 1
        self._current_memory_mb = msg.memory_mb

        # Set baseline on first job completion
        if self._baseline_memory_mb is None:
            self._baseline_memory_mb = msg.memory_mb

        logger.debug(
            "job completed in thread executor",
            extra={
                **self.logging_extra(),
                "jobs_completed": self._jobs_completed,
                "current_memory_mb": self._current_memory_mb,
                "baseline_memory_mb": self._baseline_memory_mb,
                "memory_growth_mb": self.memory_growth_mb,
            },
        )

    @utils.log_exceptions(logger=logger)
    async def _ping_task(self) -> None:
        ping_interval = utils.aio.interval(self._opts.ping_interval)
        while True:
            await ping_interval.tick()
            try:
                await channel.asend_message(self._pch, proto.PingRequest(timestamp=utils.time_ms()))
            except utils.aio.duplex_unix.DuplexClosed:
                break

    def logging_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "tid": self._thread.native_id,
        }
        if self._running_job:
            extra["job_id"] = self._running_job.job.id
            extra["room_id"] = self._running_job.job.room.sid

        return extra
