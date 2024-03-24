# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Optional,
)
from urllib.parse import urlparse

import websockets
from livekit import api, protocol
from livekit.protocol import agent as proto_agent
from livekit.protocol.agent import JobType

from .job_request import JobRequest

MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_INTERVAL = 5
ASSIGNMENT_TIMEOUT = 15

JobRequestHandler = Callable[["JobRequest"], Coroutine]


class AssignmentTimeoutError(Exception):
    """Worker timed out when joining the worker-pool"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobCancelledError(Exception):
    """Job was cancelled by the server"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class Worker:
    """A Worker is a client that connects to LiveKit Cloud (or a LiveKit server) and receives Agent jobs.
    For Job the Worker accepts, it will connect to the room and handle track subscriptions.
    """

    def __init__(
        self,
        request_handler: JobRequestHandler,
        *,
        worker_type: JobType.ValueType = JobType.JT_ROOM,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self._loop = event_loop or asyncio.get_event_loop()
        self._lock = asyncio.Lock()
        self._request_handler = request_handler
        self._worker_type = worker_type
        self._api_key = api_key or os.environ.get("LIVEKIT_API_KEY")
        self._api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        self._running = False
        self._pending_jobs: Dict[str, asyncio.Future[proto_agent.JobAssignment]] = {}

        ws_url = ws_url or os.environ.get("LIVEKIT_URL")
        if ws_url is not None:
            self._set_url(ws_url)

    def _set_url(self, ws_url: str) -> None:
        parse_res = urlparse(ws_url)
        scheme = parse_res.scheme
        if scheme.startswith("http"):
            scheme = scheme.replace("http", "ws")

        url = f"{scheme}://{parse_res.netloc}/{parse_res.path}"
        url = url.rstrip("/")

        self._agent_url = url + "/agent"
        self._rtc_url = url

    async def _connect(self) -> protocol.agent.RegisterWorkerResponse:
        self._api = api.LiveKitAPI(self._rtc_url, self._api_key, self._api_secret)

        join_jwt = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_grants(api.VideoGrants(agent=True))
            .to_jwt()
        )

        req = protocol.agent.WorkerMessage()
        req.register.type = self._worker_type

        headers = {"Authorization": f"Bearer {join_jwt}"}
        self._ws = await websockets.connect(
            self._agent_url, extra_headers=headers, close_timeout=0.150
        )
        await self._send(req)
        msg = await self._recv()
        return msg.register

    async def _send_availability(
        self, job_id: str, available: bool
    ) -> protocol.agent.JobAssignment:
        """Send availability to the server, and wait for assignment"""
        req = protocol.agent.WorkerMessage()
        req.availability.available = available
        req.availability.job_id = job_id

        f = asyncio.Future()
        self._pending_jobs[job_id] = f
        await self._send(req)

        try:
            return await asyncio.wait_for(f, ASSIGNMENT_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise AssignmentTimeoutError(
                f"assignment timeout for job {job_id}"
            ) from exc

    async def _recv(self) -> proto_agent.ServerMessage:
        message = await self._ws.recv()
        msg = protocol.agent.ServerMessage()
        msg.ParseFromString(bytes(message))  # type: ignore
        return msg

    async def _send(self, msg: protocol.agent.WorkerMessage) -> None:
        try:
            await self._ws.send(msg.SerializeToString())
        except websockets.exceptions.ConnectionClosed:
            # TODO: Implement JobStatus resuming after reconnection
            pass

    async def _handle_new_job(self, job: "JobRequest") -> None:
        try:
            await self._request_handler(job)
        except Exception:
            logging.exception("request handler for job %s failed", job.id)
            return

        if not job._answered:
            logging.warning(
                "user did not answer availability for job %s, rejecting", job.id
            )
            await job.reject()

    async def _message_received(self, msg: protocol.agent.ServerMessage) -> None:
        logging.debug("received message: %s", msg)
        which = msg.WhichOneof("message")
        if which == "availability":
            # server is asking the worker if we are available for a job
            availability = msg.availability
            job = JobRequest(self, ipc_server, availability.job)
            asyncio.ensure_future(self._handle_new_job(job), loop=self._loop)
        elif which == "assignment":
            # server is assigning a job to the worker
            assignment = msg.assignment
            job_id = assignment.job.id
            f = self._pending_jobs.get(job_id)
            if f is None:
                logging.error("received assignment for unknown job %s", job_id)
                return

            f.set_result(assignment)
            del self._pending_jobs[job_id]

    async def _reconnect(self) -> bool:
        for i in range(MAX_RECONNECT_ATTEMPTS):
            try:
                reg = await self._connect()
                logging.info("worker successfully re-registered: %s", reg.worker_id)
                return True
            except Exception as e:
                logging.error("failed to reconnect, attempt %i: %s", i, e)
                await asyncio.sleep(RECONNECT_INTERVAL)

        return False

    async def _run(self) -> None:
        try:
            while True:
                try:
                    while True:
                        await self._message_received(await self._recv())
                except websockets.exceptions.ConnectionClosed as e:
                    if self._running:
                        logging.error("connection closed, trying to reconnect: %s", e)
                        if not await self._reconnect():
                            break
                except Exception as e:
                    logging.error("error while running worker: %s", e)
                    break
        finally:
            await asyncio.shield(self._shutdown())

    async def _shutdown(self) -> None:
        async with self._lock:
            if not self._running:
                return

            await self._ws.close()
            # Close all running jobs
            await asyncio.gather(*[job.disconnect() for job in self._running_jobs])
            self._running = False

    async def start(self) -> None:
        """Start the Worker"""

        async with self._lock:
            if self._running:
                raise Exception("worker is already running")

            self._ipc_server.start()
            await self._connect()  # initial connection
            self._running = True
            self._task = self._loop.create_task(self._run())

    async def shutdown(self) -> None:
        """Shut the Worker down."""
        async with self._lock:
            if not self._running:
                return

            self._task.cancel()
            await self._task

    @property
    def id(self) -> str:
        """Worker ID"""
        return self._wid

    @property
    def running(self) -> bool:
        """Whether the worker is running.
        Running is first set to True when the websocket connection is established and
        the Worker has been acknowledged by a LiveKit Server."""
        return self._running

    @property
    def api(self) -> Optional[api.LiveKitAPI]:
        return self._api


def _run_worker(
    worker: Worker,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    started_cb: Optional[Callable[[Worker], Any]] = None,
) -> None:
    """Run the specified worker and handle graceful shutdown"""

    loop = loop or asyncio.get_event_loop()

    class GracefulShutdown(SystemExit):
        code = 1

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:

            def _signal_handler():
                raise GracefulShutdown()

            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    async def _main_task(worker: Worker) -> None:
        try:
            await worker.start()
            if started_cb:
                started_cb(worker)

            logging.info(
                "worker started, press Ctrl+C to stop (worker id: %s)", worker.id
            )

            await worker._task
        except asyncio.CancelledError:
            pass
        finally:
            logging.info("shutting down worker %s", worker.id)
            await worker.shutdown()
            logging.info("worker %s shutdown", worker.id)

    main_task = loop.create_task(_main_task(worker))
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_task)
    except (GracefulShutdown, KeyboardInterrupt):
        logging.info("Graceful shutdown worker")
    finally:
        main_task.cancel()
        loop.run_until_complete(main_task)

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()

        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)

