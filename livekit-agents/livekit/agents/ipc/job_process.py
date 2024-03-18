from __future__ import annotations

import threading
from multiprocessing import Process
from typing import Callable
from .job_main import _run_job
from livekit.protocol import agent, worker
import asyncio
from .consts import START_TIMEOUT
from ..log import worker_logger


class JobStartError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobProcess:
    def __init__(
        self,
        assignment: agent.JobAssignment,
        usercb: Callable,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._assignment = assignment
        self._send_queue = asyncio.Queue[worker.IPCWorkerMessage](32)
        self._recv_queue = asyncio.Queue[worker.IPCJobMessage](32)
        self._process = Process(target=_run_job, args=(assignment.job.id, usercb))
        self._lock = asyncio.Lock()

    def logging_extra(self) -> dict:
        pid = self._process.pid
        return {"job_id": self._assignment.job.id, "pid": pid}

    async def run(self) -> None:
        self._process.start()

        async with self._lock:
            self._started = True

        start_req = worker.StartJobRequest()
        start_req.job.CopyFrom(self._assignment.job)
        start_req.url = self._assignment.url
        start_req.token = self._assignment.token
        await self._send(worker.IPCWorkerMessage(start_job=start_req))

        # wait for the StartJobResponse
        async with asyncio.timeout(START_TIMEOUT):
            msg = await self._recv()
            if not msg.HasField("start_job"):
                raise JobStartError("unexpected response from job process")
            start_job = msg.start_job

            if start_job.error:
                raise JobStartError("error starting job process")

        # join the process & avoid zombies
        join_e = asyncio.Event()

        def _join_proc():
            self._process.join()
            self._loop.call_soon_threadsafe(join_e.set)

        join_t = threading.Thread(target=_join_proc)
        join_t.start()
        await join_e.wait()

        exitcode = self._process.exitcode
        if exitcode != 0:
            worker_logger.error(f"unexpected process exit with code {exitcode}")

    async def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown the process"""
        await self._send(worker.IPCWorkerMessage(shutdown=worker.ShutdownRequest()))
        await self._send_queue.put(None)

    async def _send(self, msg: worker.IPCWorkerMessage) -> None:
        await self._send_queue.put(msg)

    async def _recv(self) -> worker.IPCJobMessage:
        return await self._recv_queue.get()
