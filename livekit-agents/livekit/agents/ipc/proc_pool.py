from __future__ import annotations

import asyncio
from multiprocessing.context import BaseContext
from typing import Any, Awaitable, Callable, Literal

from .. import utils
from ..job import JobContext, JobExecutorType, JobProcess, RunningJobInfo
from ..log import logger
from ..utils import aio
from . import proc_job_executor, thread_job_executor
from .job_executor import JobExecutor

EventTypes = Literal[
    "process_created",
    "process_started",
    "process_ready",
    "process_closed",
    "process_job_launched",
]

MAX_CONCURRENT_INITIALIZATIONS = 1


class ProcPool(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
        job_executor_type: JobExecutorType,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._job_executor_type = job_executor_type
        self._mp_ctx = mp_ctx
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._close_timeout = close_timeout
        self._initialize_timeout = initialize_timeout
        self._loop = loop

        self._num_idle_processes = num_idle_processes
        self._init_sem = asyncio.Semaphore(MAX_CONCURRENT_INITIALIZATIONS)
        self._proc_needed_sem = asyncio.Semaphore(num_idle_processes)
        self._warmed_proc_queue = asyncio.Queue[JobExecutor]()
        self._executors: list[JobExecutor] = []
        self._started = False
        self._closed = False

    @property
    def processes(self) -> list[JobExecutor]:
        return self._executors

    def get_by_job_id(self, job_id: str) -> JobExecutor | None:
        return next(
            (
                x
                for x in self._executors
                if x.running_job and x.running_job.job.id == job_id
            ),
            None,
        )

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._main_atask = asyncio.create_task(self._main_task())

    async def aclose(self) -> None:
        if not self._started:
            return

        self._closed = True
        await aio.gracefully_cancel(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        if self._num_idle_processes == 0:
            self._proc_needed_sem.release()  # ask for a process if prewarmed processes are not disabled
            proc = await self._warmed_proc_queue.get()
        else:
            proc = await self._warmed_proc_queue.get()
            self._proc_needed_sem.release()  # notify that a new process can be warmed/started

        await proc.launch_job(info)
        self.emit("process_job_launched", proc)

    @utils.log_exceptions(logger=logger)
    async def _proc_watch_task(self) -> None:
        proc: JobExecutor
        if self._job_executor_type == JobExecutorType.THREAD:
            proc = thread_job_executor.ThreadJobExecutor(
                initialize_process_fnc=self._initialize_process_fnc,
                job_entrypoint_fnc=self._job_entrypoint_fnc,
                initialize_timeout=self._initialize_timeout,
                close_timeout=self._close_timeout,
                loop=self._loop,
            )
        elif self._job_executor_type == JobExecutorType.PROCESS:
            proc = proc_job_executor.ProcJobExecutor(
                initialize_process_fnc=self._initialize_process_fnc,
                job_entrypoint_fnc=self._job_entrypoint_fnc,
                initialize_timeout=self._initialize_timeout,
                close_timeout=self._close_timeout,
                mp_ctx=self._mp_ctx,
                loop=self._loop,
            )
        else:
            raise ValueError(f"unsupported job executor: {self._job_executor_type}")

        try:
            self._executors.append(proc)

            async with self._init_sem:
                if self._closed:
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
                except Exception:
                    self._proc_needed_sem.release()  # notify to warm a new process after initialization failure

            await proc.join()
            self.emit("process_closed", proc)
        finally:
            self._executors.remove(proc)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        watch_tasks: list[asyncio.Task[None]] = []
        try:
            while True:
                await self._proc_needed_sem.acquire()
                task = asyncio.create_task(self._proc_watch_task())
                watch_tasks.append(task)
                task.add_done_callback(watch_tasks.remove)
        except asyncio.CancelledError:
            await asyncio.gather(*[proc.aclose() for proc in self._executors])
            await asyncio.gather(*watch_tasks)
