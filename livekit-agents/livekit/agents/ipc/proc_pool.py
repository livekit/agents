from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import sys
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
        job_shutdown_fnc: Callable[[JobContext], Coroutine],
        loop: asyncio.AbstractEventLoop,
        num_idle_processes: int,
        close_timeout: float,
    ) -> None:
        super().__init__()

        if sys.platform.startswith("linux"):
            self._mp_ctx = mp.get_context("forkserver")
        else:
            self._mp_ctx = mp.get_context("spawn")

        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._job_shutdown_fnc = job_shutdown_fnc
        self._close_timeout = close_timeout
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
            job_shutdown_fnc=self._job_shutdown_fnc,
            mp_ctx=self._mp_ctx,
            loop=self._loop,
        )
        try:
            self.emit("process_created", proc)
            proc.start()
            self._processes.append(proc)
            self.emit("process_started", proc)
            await proc.initialize()
            self.emit("process_ready", proc)
            self._warmed_proc_queue.put_nowait(proc)
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
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*[proc.aclose() for proc in self._processes]),
                    timeout=self._close_timeout,
                )

            await asyncio.gather(*[proc.kill() for proc in self._processes])
            await asyncio.gather(*watch_tasks)
