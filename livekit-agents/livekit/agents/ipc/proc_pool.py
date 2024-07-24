from __future__ import annotations

import asyncio
from multiprocessing.context import BaseContext
from typing import Any, Callable, Coroutine, Literal

from .. import utils
from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from ..utils import aio
from .supervised_proc import SupervisedProc

EventTypes = Literal[
    "process_created", "process_started", "process_ready", "process_closed"
]


class ProcPool(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Coroutine],
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._mp_ctx = mp_ctx
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._close_timeout = close_timeout
        self._initialize_timeout = initialize_timeout
        self._loop = loop

        self._proc_needed_sem = asyncio.Semaphore(num_idle_processes)
        self._warmed_proc_queue = asyncio.Queue[SupervisedProc]()
        self._processes: list[SupervisedProc] = []
        self._started = False

    @property
    def processes(self) -> list[SupervisedProc]:
        return self._processes

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._main_atask = asyncio.create_task(self._main_task())

    async def aclose(self) -> None:
        if not self._started:
            return

        await aio.gracefully_cancel(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        proc = await self._warmed_proc_queue.get()
        self._proc_needed_sem.release()  # notify that a new process needs to be warmed/started
        await proc.launch_job(info)

    @utils.log_exceptions(logger=logger)
    async def _proc_watch_task(self) -> None:
        proc = SupervisedProc(
            initialize_process_fnc=self._initialize_process_fnc,
            job_entrypoint_fnc=self._job_entrypoint_fnc,
            initialize_timeout=self._initialize_timeout,
            close_timeout=self._close_timeout,
            mp_ctx=self._mp_ctx,
            loop=self._loop,
        )
        try:
            self.emit("process_created", proc)
            proc.start()
            self._processes.append(proc)
            self.emit("process_started", proc)
            try:
                await proc.initialize()
                # process where initialization times out will never fire "process_ready"
                # neither be used to launch jobs
                self.emit("process_ready", proc)
                self._warmed_proc_queue.put_nowait(proc)
            except asyncio.TimeoutError:
                self._proc_needed_sem.release()  # notify to warm a new process after initialization failure
            await proc.join()
            self.emit("process_closed", proc)
        finally:
            self._processes.remove(proc)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        watch_tasks = []
        try:
            while True:
                await self._proc_needed_sem.acquire()
                task = asyncio.create_task(self._proc_watch_task())
                watch_tasks.append(task)
                task.add_done_callback(watch_tasks.remove)
        except asyncio.CancelledError:
            await asyncio.gather(*[proc.aclose() for proc in self._processes])
            await asyncio.gather(*watch_tasks)
