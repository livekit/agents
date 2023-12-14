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

import asyncio
import logging
import os
import signal
import uuid
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Optional,
    Tuple,
)
from urllib.parse import urlparse

import websockets

from livekit import api, protocol
from livekit.protocol import agent as proto_agent
from livekit.protocol import models as proto_models
from livekit.protocol.agent import JobType as ProtoJobType
from .job_request import JobRequest
from .job_context import JobContext

MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_INTERVAL = 5
ASSIGNMENT_TIMEOUT = 15

JobType = ProtoJobType


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
        job_request_cb: Callable[["JobRequest"], Coroutine],
        worker_type: JobType.ValueType,
        *,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        ws_url: str = os.environ.get("LIVEKIT_URL", "http://localhost:7880"),
        api_key: str = os.environ.get("LIVEKIT_API_KEY", ""),
        api_secret: str = os.environ.get("LIVEKIT_API_SECRET", ""),
    ) -> None:
        """
        Args:
            job_request_cb (Callable[[JobRequest], Coroutine]): Callback that is triggered when a new Job is available.
            worker_type (JobType.ValueType): What kind of jobs this worker can handle.
            event_loop (Optional[asyncio.AbstractEventLoop], optional): Optional asyncio event loop to use for this worker. Defaults to None.
            ws_url (_type_, optional): LiveKit websocket URL. Defaults to os.environ.get("LIVEKIT_URL", "http://localhost:7880").
            api_key (str, optional): LiveKit API Key. Defaults to os.environ.get("LIVEKIT_API_KEY", "").
            api_secret (str, optional): LiveKit API Secret. Defaults to os.environ.get("LIVEKIT_API_SECRET", "").
        """
        self._set_url(ws_url)

        self._loop = event_loop or asyncio.get_event_loop()
        self._lock = asyncio.Lock()
        self._job_request_cb = job_request_cb
        self._wid = "W-" + str(uuid.uuid4())[:12]
        self._worker_type = worker_type
        self._api_key = api_key
        self._api_secret = api_secret
        self._running = False
        self._running_jobs: list["JobContext"] = []
        self._pending_jobs: Dict[str, asyncio.Future[proto_agent.JobAssignment]] = {}

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
        join_jwt = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_grants(api.VideoGrants(agent=True))
            .to_jwt()
        )

        req = protocol.agent.WorkerMessage()
        req.register.worker_id = self._wid
        req.register.type = self._worker_type

        headers = {"Authorization": f"Bearer {join_jwt}"}
        self._ws = await websockets.connect(
            self._agent_url, extra_headers=headers, close_timeout=0.150
        )
        await self._send(req)
        res = await self._recv()
        return res.register

    async def _send_availability(
        self, job_id: str, available: bool
    ) -> protocol.agent.JobAssignment:
        """
        Send availability to the server, and wait for assignment
        """
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

    async def _send_job_status(
        self,
        job_id: str,
        status: protocol.agent.JobStatus.ValueType,
        error: str,
        user_data: str = "",
    ) -> None:
        req = protocol.agent.WorkerMessage()
        req.job_update.job_id = job_id
        req.job_update.status = status
        req.job_update.error = error
        req.job_update.user_data = user_data
        await self._ws.send(req.SerializeToString())

    def _simulate_job(
        self,
        room: proto_models.Room,
        participant: Optional[proto_models.ParticipantInfo],
    ):
        # TODO(theomonnom): the server could handle the JobSimulation like
        # we're doing with the SFU today
        job_id = "JR_" + str(uuid.uuid4())[:12]
        job_type = JobType.JT_ROOM if participant is None else JobType.JT_PUBLISHER
        job = proto_agent.Job(
            id=job_id, type=job_type, room=room, participant=participant
        )
        job = JobRequest(self, job, simulated=True)
        asyncio.ensure_future(self._handle_new_job(job), loop=self._loop)

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
        """
        Execute the available callback, and automatically deny the job if the callback
        does not send an answer or raises an exception
        """

        try:
            await self._job_request_cb(job)
        except Exception as e:
            logging.error("available callback failed: %s", e)
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
            job = JobRequest(self, availability.job)
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
        """
        Whether the worker is running.
        Running is first set to True when the websocket connection is established and
        the Worker has been acknowledged by a LiveKit Server.
        """
        return self._running


def _run_worker(
    worker: Worker,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    started_cb: Optional[Callable[[Worker], Any]] = None,
) -> None:
    """
    Run the specified worker and handle graceful shutdown
    """

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


def run_app(worker: Worker) -> None:
    """
    Run the CLI to interact with the worker
    """

    import click

    @click.group()
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        help="Set the logging level",
    )
    @click.option(
        "--url",
        help="The websocket URL",
        default=worker._rtc_url,
    )
    @click.option("--api-key", help="The API key", default=worker._api_key)
    @click.option("--api-secret", help="The API secret", default=worker._api_secret)
    def cli(log_level: str, url: str, api_key: str, api_secret: str) -> None:
        logging.basicConfig(level=log_level)
        worker._set_url(url)
        worker._api_key = api_key
        worker._api_secret = api_secret

    @cli.command(help="Start the worker")
    def start() -> None:
        _run_worker(worker)

    @cli.command(help="Start a worker and simulate a job, useful for testing")
    @click.option("--room-name", help="The room name", required=True)
    @click.option("--identity", help="The participant identity")
    def simulate_job(room_name: str, identity: str) -> None:
        async def _pre_run() -> (
            Tuple[proto_models.Room, Optional[proto_models.ParticipantInfo]]
        ):
            lkapi = api.LiveKitAPI(worker._rtc_url, worker._api_key, worker._api_secret)

            try:
                room = await lkapi.room.create_room(
                    api.CreateRoomRequest(name=room_name)
                )

                participant = None
                if identity:
                    participant = await lkapi.room.get_participant(
                        api.RoomParticipantIdentity(room=room_name, identity=identity)
                    )

                return room, participant
            finally:
                await lkapi.aclose()

        room_info, participant = worker._loop.run_until_complete(_pre_run())
        logging.info(f"Simulating job for room {room_info.name} ({room_info.sid})")
        _run_worker(
            worker, started_cb=lambda _: worker._simulate_job(room_info, participant)
        )

    cli()
