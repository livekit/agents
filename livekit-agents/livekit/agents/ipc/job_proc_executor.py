from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import socket
from collections.abc import Awaitable
from multiprocessing.context import BaseContext
from typing import Any, Callable

from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from ..telemetry import metrics
from ..utils import aio, log_exceptions, shortuuid
from ..utils.aio import duplex_unix
from . import channel, proto
from .inference_executor import InferenceExecutor
from .job_executor import JobStatus
from .job_proc_lazy_main import ProcStartArgs, proc_main
from .supervised_proc import SupervisedProc


class ProcJobExecutor(SupervisedProc):
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        inference_executor: InferenceExecutor | None,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
        http_proxy: str | None,
        reuse_process: bool,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            memory_warn_mb=memory_warn_mb,
            memory_limit_mb=memory_limit_mb,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            high_ping_threshold=high_ping_threshold,
            mp_ctx=mp_ctx,
            loop=loop,
            http_proxy=http_proxy,
        )

        self._user_args: Any | None = None
        self._job_status: JobStatus | None = None
        self._running_job: RunningJobInfo | None = None
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._session_end_fnc = session_end_fnc
        self._inference_executor = inference_executor
        self._inference_tasks: list[asyncio.Task[None]] = []
        self._id = shortuuid("PCEXEC_")
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

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        levels = {}
        root = logging.getLogger()
        levels["root"] = root.level
        children = logging.Logger.manager.loggerDict.values()
        for child in children:
            if isinstance(child, logging.Logger):
                levels[child.name] = child.level

        proc_args = ProcStartArgs(
            initialize_process_fnc=self._initialize_process_fnc,
            job_entrypoint_fnc=self._job_entrypoint_fnc,
            session_end_fnc=self._session_end_fnc,
            log_cch=log_cch,
            mp_cch=cch,
            user_arguments=self._user_args,
            logger_levels=levels,
            reuse_process=self._reuse_process,
        )

        return self._mp_ctx.Process(  # type: ignore
            target=proc_main, args=(proc_args,), name="job_proc"
        )

    @log_exceptions(logger=logger)
    async def _main_task(self, ipc_ch: aio.ChanReceiver[channel.Message]) -> None:
        try:
            async for msg in ipc_ch:
                if isinstance(msg, proto.InferenceRequest):
                    self._inference_tasks.append(asyncio.create_task(self._do_inference_task(msg)))
                elif isinstance(msg, proto.JobCompleted):
                    self._handle_job_completed(msg)
        finally:
            await aio.cancel_and_wait(*self._inference_tasks)

    def _handle_job_completed(self, msg: proto.JobCompleted) -> None:
        """Handle job completion notification from subprocess"""
        self._jobs_completed += 1
        self._current_memory_mb = msg.memory_mb

        # Set baseline on first job completion
        if self._baseline_memory_mb is None:
            self._baseline_memory_mb = msg.memory_mb

        # Clean up completed inference tasks to avoid list growth
        self._inference_tasks = [t for t in self._inference_tasks if not t.done()]

        logger.debug(
            "job completed in process",
            extra={
                **self.logging_extra(),
                "jobs_completed": self._jobs_completed,
                "current_memory_mb": self._current_memory_mb,
                "baseline_memory_mb": self._baseline_memory_mb,
                "memory_growth_mb": self.memory_growth_mb,
            },
        )

    @log_exceptions(logger=logger)
    async def _supervise_task(self) -> None:
        try:
            await super()._supervise_task()
        finally:
            if self._running_job:
                metrics.job_ended()
                self._job_status = JobStatus.SUCCESS if self.exitcode == 0 else JobStatus.FAILED

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
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")

        if not self._initialize_fut.done():
            raise RuntimeError("process not initialized")

        # Reset state for this new job
        self._job_reused_event.clear()
        self._closing = False

        metrics.job_started()
        self._job_status = JobStatus.RUNNING
        self._running_job = info

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
        Attempt to gracefully close the process.
        If process reuse is enabled and job completes during shutdown, skip killing.
        """
        if not self.started:
            return

        # Prevent concurrent close attempts
        if self._closing:
            # Already closing, just wait for completion
            async with self._lock:
                if self._supervise_atask:
                    await asyncio.shield(self._supervise_atask)
            return

        self._closing = True
        has_running_job = self._running_job is not None

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await channel.asend_message(self._pch, proto.ShutdownRequest())

        # Wait for either process exit OR job reuse (if reuse enabled with active job)
        try:
            if self._supervise_atask:
                if self._reuse_process and has_running_job:
                    # Create a task for the reuse event
                    reuse_task = asyncio.create_task(self._job_reused_event.wait())

                    # Wait for either supervise task completion or job reuse event
                    done, pending = await asyncio.wait(
                        [self._supervise_atask, reuse_task],
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
                            "process completed job and will be reused, skipping shutdown",
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
                        asyncio.shield(self._supervise_atask), timeout=self._opts.close_timeout
                    )
        except asyncio.TimeoutError:
            logger.error(
                "process did not exit in time, killing process",
                extra=self.logging_extra(),
            )
            await self._send_dump_signal()
            await self._send_kill_signal()

        async with self._lock:
            if self._supervise_atask:
                await asyncio.shield(self._supervise_atask)

    def logging_extra(self) -> dict[str, Any]:
        extra = super().logging_extra()

        if self._running_job:
            extra["job_id"] = self._running_job.job.id
            extra["room_id"] = self._running_job.job.room.sid

        return extra
