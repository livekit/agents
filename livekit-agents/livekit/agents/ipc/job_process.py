from __future__ import annotations

from multiprocessing import Process
from .job_main import _run_job
from livekit.protocol import agent, worker

import logging
import asyncio

START_TIMEOUT = 5


class JobStartError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobProcess:
    def __init__(
        self,
        assignment: agent.JobAssignment,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._assignment = assignment
        self._send_queue = asyncio.Queue[worker.WorkerMessage | None](32)
        self._recv_queue = asyncio.Queue[worker.JobMessage](32)
        self._process = Process(target=_run_job, args=(self._send_queue,))

    async def start(self) -> None:
        """Start the process"""
        self._process.start()

        start_req = worker.StartJobRequest()
        start_req.job = self._assignment.job
        start_req.url = self._assignment.url
        start_req.token = self._assignment.token
        await self._send(worker.WorkerMessage(start_job=start_req))

        # wait for the StartJobResponse
        async with asyncio.timeout(START_TIMEOUT):
            msg = await self._recv()
            if not msg.HasField("start_job"):
                raise JobStartError("unexpected response from job process")
            start_job = msg.start_job

            if start_job.error:
                raise JobStartError("error starting job process")

        # job started successfully
        logging.info(f"process started job: {self._assignment.job.job_id}")

    async def _shutdown(self) -> None:
        """Gracefully shutdown the jop"""
        await self._send(worker.WorkerMessage(shutdown=worker.ShutdownRequest()))
        await self._send_queue.put(None)
        await self._loop.run_in_executor(None, self._process.join)

    async def _send(self, msg: worker.WorkerMessage) -> None:
        await self._send_queue.put(msg)

    async def _recv(self) -> worker.JobMessage:
        return await self._recv_queue.get()
