from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from multiprocessing.context import BaseContext
from typing import Any, Literal

from .. import _core, utils
from ..job import JobContext, JobExecutorType, JobProcess, RunningJobInfo
from ..log import logger
from . import inference_executor, job_proc_executor, job_thread_executor
from .job_executor import JobExecutor

EventTypes = Literal[
    "process_created",
    "process_started",
    "process_ready",
    "process_closed",
    "process_job_launched",
]


class ProcPool(utils.EventEmitter[EventTypes]):
    """The warm job processes. The core keeps ``num_idle_processes`` warm, hands a job the next
    one and retries a failed spawn or launch (``_core.ProcPool``); this class makes the
    executors."""

    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
        session_end_timeout: float,
        inference_executor: inference_executor.InferenceExecutor | None,
        job_executor_type: JobExecutorType,
        mp_ctx: BaseContext,
        memory_warn_mb: float,
        memory_limit_mb: float,
        http_proxy: str | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._job_executor_type = job_executor_type
        self._executor_args: dict[str, Any] = {
            "initialize_process_fnc": initialize_process_fnc,
            "job_entrypoint_fnc": job_entrypoint_fnc,
            "session_end_fnc": session_end_fnc,
            "simulation_end_fnc": simulation_end_fnc,
            "initialize_timeout": initialize_timeout,
            "close_timeout": close_timeout,
            "session_end_timeout": session_end_timeout,
            "inference_executor": inference_executor,
            "ping_interval": 2.5,
            "high_ping_threshold": 0.5,
            "http_proxy": http_proxy,
            "loop": loop,
        }
        self._proc_args: dict[str, Any] = {
            "mp_ctx": mp_ctx,
            "ping_timeout": 60,
            "memory_warn_mb": memory_warn_mb,
            "memory_limit_mb": memory_limit_mb,
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

    async def launch_job(self, info: RunningJobInfo) -> None:
        await self._pool.launch_job(info)

    def set_target_idle_processes(self, num_idle_processes: int) -> None:
        self._pool.target_idle_processes = num_idle_processes

    @property
    def target_idle_processes(self) -> int:
        return self._pool.target_idle_processes  # type: ignore[no-any-return]

    def _core_create(self) -> JobExecutor:
        if self._job_executor_type == JobExecutorType.THREAD:
            return job_thread_executor.ThreadJobExecutor(**self._executor_args)
        if self._job_executor_type == JobExecutorType.PROCESS:
            return job_proc_executor.ProcJobExecutor(**self._executor_args, **self._proc_args)
        raise ValueError(f"unsupported job executor: {self._job_executor_type}")

    def _core_init_failed(self, proc: JobExecutor, error: BaseException) -> None:
        logger.error("error initializing process", extra=proc.logging_extra(), exc_info=error)
