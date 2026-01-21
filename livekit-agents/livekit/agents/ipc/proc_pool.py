from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable
from multiprocessing.context import BaseContext
from typing import Any, Callable, Literal

from .. import utils
from ..job import JobContext, JobExecutorType, JobProcess, RunningJobInfo
from ..log import logger
from ..utils import aio
from ..utils.hw.cpu import get_cpu_monitor
from . import inference_executor, job_proc_executor, job_thread_executor
from .job_executor import JobExecutor

EventTypes = Literal[
    "process_created",
    "process_started",
    "process_ready",
    "process_closed",
    "process_job_launched",
]

MAX_CONCURRENT_INITIALIZATIONS = min(math.ceil(get_cpu_monitor().cpu_count()), 4)


class ProcPool(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
        inference_executor: inference_executor.InferenceExecutor | None,
        job_executor_type: JobExecutorType,
        mp_ctx: BaseContext,
        memory_warn_mb: float,
        memory_limit_mb: float,
        reuse_processes: bool,
        max_process_reuses: int,
        reuse_memory_growth_mb: float,
        max_idle_processes: int | None,
        http_proxy: str | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._job_executor_type = job_executor_type
        self._mp_ctx = mp_ctx
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._session_end_fnc = session_end_fnc
        self._close_timeout = close_timeout
        self._inf_executor = inference_executor
        self._initialize_timeout = initialize_timeout
        self._loop = loop
        self._memory_limit_mb = memory_limit_mb
        self._memory_warn_mb = memory_warn_mb
        self._default_num_idle_processes = num_idle_processes
        self._http_proxy = http_proxy
        self._target_idle_processes = num_idle_processes
        self._reuse_processes = reuse_processes and job_executor_type == JobExecutorType.PROCESS
        self._max_process_reuses = max_process_reuses
        self._reuse_memory_growth_mb = reuse_memory_growth_mb
        self._max_idle_processes = max_idle_processes or (num_idle_processes * 2)

        self._init_sem = asyncio.Semaphore(MAX_CONCURRENT_INITIALIZATIONS)
        self._warmed_proc_queue = asyncio.Queue[JobExecutor]()
        self._executors: list[JobExecutor] = []
        self._spawn_tasks: set[asyncio.Task[None]] = set()
        self._monitor_tasks: set[asyncio.Task[None]] = set()
        self._started = False
        self._closed = False

        self._idle_ready = asyncio.Event()
        self._jobs_waiting_for_process = 0

    @property
    def processes(self) -> list[JobExecutor]:
        return self._executors

    def get_by_job_id(self, job_id: str) -> JobExecutor | None:
        return next(
            (x for x in self._executors if x.running_job and x.running_job.job.id == job_id),
            None,
        )

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._main_atask = asyncio.create_task(self._main_task())

        if self._default_num_idle_processes > 0:
            # wait for the idle processes to be warmed up (by the main task)
            await self._idle_ready.wait()

    async def aclose(self) -> None:
        if not self._started:
            return

        self._closed = True
        await aio.cancel_and_wait(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        self._jobs_waiting_for_process += 1
        if (
            self._warmed_proc_queue.empty()
            and len(self._spawn_tasks) < self._jobs_waiting_for_process
        ):
            # spawn a new process if there are no idle processes
            task = asyncio.create_task(self._proc_spawn_task())
            self._spawn_tasks.add(task)
            task.add_done_callback(self._spawn_tasks.discard)

        proc = await self._warmed_proc_queue.get()
        self._jobs_waiting_for_process -= 1

        # Track initial job count before launch
        initial_job_count = proc.jobs_completed

        await proc.launch_job(info)
        self.emit("process_job_launched", proc)

        # Monitor job completion for reuse
        if self._reuse_processes:
            monitor_task = asyncio.create_task(
                self._monitor_job_completion(proc, initial_job_count)
            )
            self._monitor_tasks.add(monitor_task)
            monitor_task.add_done_callback(self._monitor_tasks.discard)

    @utils.log_exceptions(logger=logger)
    async def _monitor_job_completion(self, proc: JobExecutor, initial_job_count: int) -> None:
        """Monitor a job for completion to decide whether to reuse the process"""
        # Wait for the job to complete (jobs_completed counter increases)
        while proc.jobs_completed == initial_job_count and proc.started:
            await asyncio.sleep(0.1)

        if not proc.started or self._closed:
            return

        # Check if process can be reused
        can_reuse = self._can_reuse_process(proc)

        if can_reuse:
            # Check if we need to manage idle process limit
            idle_count = self._warmed_proc_queue.qsize()
            if idle_count >= self._max_idle_processes:
                # Close the most-used idle process to make room
                await self._close_most_used_idle_process()

            logger.info(
                "reusing process",
                extra={
                    **proc.logging_extra(),
                    "jobs_completed": proc.jobs_completed,
                    "memory_growth_mb": proc.memory_growth_mb,
                    "idle_processes": idle_count,
                },
            )
            # Clear job state and return to pool
            proc.clear_running_job()
            self._warmed_proc_queue.put_nowait(proc)
        else:
            logger.info(
                "retiring process after job completion",
                extra={
                    **proc.logging_extra(),
                    "jobs_completed": proc.jobs_completed,
                    "memory_growth_mb": proc.memory_growth_mb,
                    "reason": self._get_retire_reason(proc),
                },
            )
            # Clear the running job state so the process can receive shutdown
            proc.clear_running_job()
            # Shutdown and remove from pool - the process is idle and will exit quickly
            await proc.aclose()

    def _can_reuse_process(self, proc: JobExecutor) -> bool:
        """Determine if a process can be reused based on health checks"""
        if not self._reuse_processes:
            return False

        # Check reuse count limit
        if self._max_process_reuses > 0 and proc.jobs_completed >= self._max_process_reuses:
            return False

        # Check memory growth
        if (
            self._reuse_memory_growth_mb > 0
            and proc.memory_growth_mb > self._reuse_memory_growth_mb
        ):
            return False

        return True

    def _get_retire_reason(self, proc: JobExecutor) -> str:
        """Get the reason why a process is being retired"""
        if not self._reuse_processes:
            return "reuse disabled"

        if self._max_process_reuses > 0 and proc.jobs_completed >= self._max_process_reuses:
            return f"max reuses reached ({proc.jobs_completed}/{self._max_process_reuses})"

        if (
            self._reuse_memory_growth_mb > 0
            and proc.memory_growth_mb > self._reuse_memory_growth_mb
        ):
            return f"memory growth exceeded ({proc.memory_growth_mb:.1f}MB > {self._reuse_memory_growth_mb}MB)"

        return "unknown"

    async def _close_most_used_idle_process(self) -> None:
        """Close the most-used idle process when the idle limit is exceeded"""
        # Drain all idle processes from the queue
        idle_procs: list[JobExecutor] = []
        while not self._warmed_proc_queue.empty():
            try:
                proc = self._warmed_proc_queue.get_nowait()
                idle_procs.append(proc)
            except asyncio.QueueEmpty:
                break

        if not idle_procs:
            return

        # Sort by jobs_completed (descending) to find the most-used process
        idle_procs.sort(key=lambda p: p.jobs_completed, reverse=True)
        most_used = idle_procs[0]

        logger.info(
            "closing most-used idle process due to max idle limit",
            extra={
                **most_used.logging_extra(),
                "jobs_completed": most_used.jobs_completed,
                "max_idle_processes": self._max_idle_processes,
            },
        )

        # Close the most-used process
        asyncio.create_task(most_used.aclose())

        # Put the rest back in the queue
        for proc in idle_procs[1:]:
            self._warmed_proc_queue.put_nowait(proc)

    def set_target_idle_processes(self, num_idle_processes: int) -> None:
        self._target_idle_processes = num_idle_processes

    @property
    def target_idle_processes(self) -> int:
        return self._target_idle_processes

    @utils.log_exceptions(logger=logger)
    async def _proc_spawn_task(self) -> None:
        proc: JobExecutor
        if self._job_executor_type == JobExecutorType.THREAD:
            proc = job_thread_executor.ThreadJobExecutor(
                initialize_process_fnc=self._initialize_process_fnc,
                job_entrypoint_fnc=self._job_entrypoint_fnc,
                session_end_fnc=self._session_end_fnc,
                initialize_timeout=self._initialize_timeout,
                close_timeout=self._close_timeout,
                inference_executor=self._inf_executor,
                ping_interval=2.5,
                high_ping_threshold=0.5,
                http_proxy=self._http_proxy,
                loop=self._loop,
            )
        elif self._job_executor_type == JobExecutorType.PROCESS:
            proc = job_proc_executor.ProcJobExecutor(
                initialize_process_fnc=self._initialize_process_fnc,
                job_entrypoint_fnc=self._job_entrypoint_fnc,
                session_end_fnc=self._session_end_fnc,
                initialize_timeout=self._initialize_timeout,
                close_timeout=self._close_timeout,
                inference_executor=self._inf_executor,
                mp_ctx=self._mp_ctx,
                loop=self._loop,
                ping_interval=2.5,
                ping_timeout=60,
                high_ping_threshold=0.5,
                memory_warn_mb=self._memory_warn_mb,
                memory_limit_mb=self._memory_limit_mb,
                http_proxy=self._http_proxy,
                reuse_process=self._reuse_processes,
            )
        else:
            raise ValueError(f"unsupported job executor: {self._job_executor_type}")

        self._executors.append(proc)
        async with self._init_sem:
            if self._closed:
                self._executors.remove(proc)
                return

            self.emit("process_created", proc)
            await proc.start()
            self.emit("process_started", proc)
            try:
                await proc.initialize()
                # process where initialization times out will never fire "process_ready"
                # neither be used to launch jobs

                self.emit("process_ready", proc)
                self._warmed_proc_queue.put_nowait(proc)
                if self._warmed_proc_queue.qsize() >= self._default_num_idle_processes:
                    self._idle_ready.set()
            except Exception:
                logger.exception("error initializing process", extra=proc.logging_extra())

        monitor_task = asyncio.create_task(self._monitor_process_task(proc))
        self._monitor_tasks.add(monitor_task)
        monitor_task.add_done_callback(self._monitor_tasks.discard)

    @utils.log_exceptions(logger=logger)
    async def _monitor_process_task(self, proc: JobExecutor) -> None:
        """Monitor process lifecycle - this tracks when a process exits completely"""
        try:
            await proc.join()
            self.emit("process_closed", proc)
        finally:
            if proc in self._executors:
                self._executors.remove(proc)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            while not self._closed:
                current_pending = self._warmed_proc_queue.qsize() + len(self._spawn_tasks)
                to_spawn = (
                    min(self._target_idle_processes, self._default_num_idle_processes)
                    - current_pending
                )

                for _ in range(to_spawn):
                    task = asyncio.create_task(self._proc_spawn_task())
                    self._spawn_tasks.add(task)
                    task.add_done_callback(self._spawn_tasks.discard)

                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            await asyncio.gather(*[proc.aclose() for proc in self._executors])
            await asyncio.gather(*self._spawn_tasks)
            await asyncio.gather(*self._monitor_tasks)
