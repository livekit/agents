from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
from multiprocessing.context import BaseContext
from typing import Any, Awaitable, Callable

from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from ..utils import aio, log_exceptions
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
        inference_executor: InferenceExecutor | None,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
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
        )

        self._user_args: Any | None = None
        self._job_status: JobStatus | None = None
        self._running_job: RunningJobInfo | None = None
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._inference_executor = inference_executor
        self._inference_tasks: list[asyncio.Task[None]] = []

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

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        proc_args = ProcStartArgs(
            initialize_process_fnc=self._initialize_process_fnc,
            job_entrypoint_fnc=self._job_entrypoint_fnc,
            log_cch=log_cch,
            mp_cch=cch,
            user_arguments=self._user_args,
        )

        return self._mp_ctx.Process(  # type: ignore
            target=proc_main,
            args=(proc_args,),
            name="job_proc",
        )

    @log_exceptions(logger=logger)
    async def _main_task(self, ipc_ch: aio.ChanReceiver[channel.Message]) -> None:
        try:
            async for msg in ipc_ch:
                if isinstance(msg, proto.InferenceRequest):
                    self._inference_tasks.append(
                        asyncio.create_task(self._do_inference_task(msg))
                    )
        finally:
            await aio.gracefully_cancel(*self._inference_tasks)

    @log_exceptions(logger=logger)
    async def _supervise_task(self) -> None:
        try:
            await super()._supervise_task()
        finally:
            self._job_status = (
                JobStatus.SUCCESS if self.exitcode == 0 else JobStatus.FAILED
            )

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
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")

        if not self._initialize_fut.done():
            raise RuntimeError("process not initialized")

        self._job_status = JobStatus.RUNNING
        self._running_job = info

        start_req = proto.StartJobRequest()
        start_req.running_job = info
        await channel.asend_message(self._pch, start_req)

    def logging_extra(self):
        extra = super().logging_extra()

        if self._running_job:
            extra["job_id"] = self._running_job.job.id

        return extra
