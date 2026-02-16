from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from multiprocessing.context import BaseContext
from typing import Any, Literal

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

        self._init_sem = asyncio.Semaphore(MAX_CONCURRENT_INITIALIZATIONS)
        self._warmed_proc_queue = asyncio.Queue[JobExecutor]()
        self._executors: list[JobExecutor] = []
        self._spawn_tasks: set[asyncio.Task[None]] = set()
        self._close_tasks: set[asyncio.Task[None]] = set()
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
            # use a timeout so start() doesn't block forever if initialization fails
            try:
                await asyncio.wait_for(
                    self._idle_ready.wait(),
                    timeout=self._initialize_timeout + 2,
                )
            except asyncio.TimeoutError:
                logger.warning("timed out waiting for idle processes to initialize")

    async def aclose(self) -> None:
        if not self._started:
            return

        self._closed = True
        await aio.cancel_and_wait(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        MAX_ATTEMPTS = 3

        for attempt in range(MAX_ATTEMPTS):
            self._jobs_waiting_for_process += 1
            try:
                if (
                    self._warmed_proc_queue.empty()
                    and len(self._spawn_tasks) < self._jobs_waiting_for_process
                ):
                    # spawn a new process if there are no idle processes
                    task = asyncio.create_task(self._proc_spawn_task())
                    self._spawn_tasks.add(task)
                    task.add_done_callback(self._spawn_tasks.discard)

                if self._warmed_proc_queue.empty():
                    logger.warning(
                        "no warmed process available for job, waiting for one to be created",
                        extra={"job_id": info.job.id},
                    )

                proc = await self._warmed_proc_queue.get()
            finally:
                self._jobs_waiting_for_process -= 1

            try:
                await proc.launch_job(info)
                self.emit("process_job_launched", proc)
                return
            except Exception:
                close_task = asyncio.create_task(proc.aclose())
                self._close_tasks.add(close_task)
                close_task.add_done_callback(self._close_tasks.discard)
                if attempt == MAX_ATTEMPTS - 1:
                    logger.error(
                        "failed to launch job on process after %d attempts",
                        MAX_ATTEMPTS,
                        extra={"job_id": info.job.id},
                    )
                    raise
                logger.warning(
                    "failed to launch job on process, retrying with a new process",
                    extra={"job_id": info.job.id, "attempt": attempt + 1},
                )

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
            )
        else:
            raise ValueError(f"unsupported job executor: {self._job_executor_type}")

        self._executors.append(proc)
        initialized = False
        try:
            async with self._init_sem:
                if not self._closed:
                    self.emit("process_created", proc)
                    await proc.start()
                    self.emit("process_started", proc)
                    await proc.initialize()
                    self.emit("process_ready", proc)
                    self._warmed_proc_queue.put_nowait(proc)
                    if self._warmed_proc_queue.qsize() >= self._default_num_idle_processes:
                        self._idle_ready.set()

                    initialized = True
        except Exception:
            logger.exception("error initializing process", extra=proc.logging_extra())
        except asyncio.CancelledError:
            pass

        if not initialized:
            self._executors.remove(proc)
            await proc.aclose()
            self.emit("process_closed", proc)
            return

        monitor_task = asyncio.create_task(self._monitor_process_task(proc))
        self._monitor_tasks.add(monitor_task)
        monitor_task.add_done_callback(self._monitor_tasks.discard)

    @utils.log_exceptions(logger=logger)
    async def _monitor_process_task(self, proc: JobExecutor) -> None:
        try:
            await proc.join()
            self.emit("process_closed", proc)
        finally:
            self._executors.remove(proc)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            while not self._closed:
                current_pending = self._warmed_proc_queue.qsize() + len(self._spawn_tasks)
                target = max(
                    min(self._target_idle_processes, self._default_num_idle_processes),
                    self._jobs_waiting_for_process,
                )
                to_spawn = target - current_pending

                for _ in range(to_spawn):
                    task = asyncio.create_task(self._proc_spawn_task())
                    self._spawn_tasks.add(task)
                    task.add_done_callback(self._spawn_tasks.discard)

                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            await aio.cancel_and_wait(*self._spawn_tasks)
            await asyncio.gather(*[proc.aclose() for proc in self._executors])
            await asyncio.gather(*self._close_tasks)
            await asyncio.gather(*self._monitor_tasks)
