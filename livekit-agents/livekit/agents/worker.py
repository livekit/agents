# Copyright 2024 LiveKit, Inc.
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

import logging
import uuid
import os
import asyncio
import signal
from typing import (
    Any,
    Coroutine,
    Dict,
    Optional,
    Callable,
    Tuple,
)

from click import Option
from websockets import frames
from livekit.protocol import agent as proto_agent
from livekit.protocol import models as proto_models
from livekit.plugins.core import Plugin
from dataclasses import dataclass
from urllib.parse import urlparse
from contextlib import aclosing

import websockets
from livekit import api, rtc, protocol

MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_INTERVAL = 5
ASSIGNMENT_TIMEOUT = 15


def subscribe_all(*args) -> bool:
    return True


class AssignmentTimeoutError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobCancelledError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Worker:
    def __init__(
        self,
        available_cb: Callable[["JobRequest"], Coroutine],
        worker_type: protocol.agent.JobType.ValueType,
        *,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        ws_url: str = os.environ.get("LIVEKIT_URL", "http://localhost:7880"),
        api_key: str = os.environ.get("LIVEKIT_API_KEY", ""),
        api_secret: str = os.environ.get("LIVEKIT_API_SECRET", ""),
    ) -> None:
        self._set_url(ws_url)

        self._loop = event_loop or asyncio.get_event_loop()
        self._lock = asyncio.Lock()
        self._available_cb = available_cb
        self._wid = "W-" + str(uuid.uuid4())[:12]
        self._worker_type = worker_type
        self._api_key = api_key
        self._api_secret = api_secret
        self._running = False
        self._running_jobs: list["JobContext"] = []
        self._pending_jobs: Dict[str,
                                 asyncio.Future[proto_agent.JobAssignment]] = {}

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
        self._ws = await websockets.connect(self._agent_url, extra_headers=headers, close_timeout=0.150)
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
        except asyncio.TimeoutError:
            raise AssignmentTimeoutError(
                f"assignment timeout for job {job_id}")

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

    def _simulate_job(self, room: proto_models.Room, participant: Optional[proto_models.ParticipantInfo]):
        # TODO(theomonnom): the server could handle the JobSimulation like we're doing with the SFU today
        job_id = "JR_" + str(uuid.uuid4())[:12]
        job_type = proto_agent.JobType.JT_ROOM if participant is None else proto_agent.JobType.JT_PUBLISHER
        job = proto_agent.Job(id=job_id, type=job_type,
                              room=room, participant=participant)
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
            await self._available_cb(job)
        except Exception as e:
            logging.error(f"available callback failed: {e}")
            return

        if not job._answered:
            logging.warn(
                f"user did not answer availability for job {job.id}, rejecting"
            )
            await job.reject()

    async def _message_received(self, msg: protocol.agent.ServerMessage) -> None:
        logging.debug(f"received message: {msg}")
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
                logging.error(f"received assignment for unknown job {job_id}")
                return

            f.set_result(assignment)
            del self._pending_jobs[job_id]

    async def _reconnect(self) -> bool:
        for i in range(MAX_RECONNECT_ATTEMPTS):
            try:
                reg = await self._connect()
                logging.info(
                    f"worker successfully re-registered: {reg.worker_id}")
                return True
            except Exception as e:
                logging.error(f"failed to reconnect, attempt {i}: {e}")
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
                        logging.error(
                            f"connection closed, trying to reconnect: {e}")
                        if not await self._reconnect():
                            break
                except Exception as e:
                    logging.error(f"error while running worker: {e}")
                    break

        except asyncio.CancelledError:
            await self._ws.close()

    @property
    def id(self) -> str:
        return self._wid

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        async with self._lock:
            if self._running:
                raise Exception("worker is already running")

            await self._connect()  # initial connection
            self._running = True
            self._task = self._loop.create_task(self._run())

    async def shutdown(self) -> None:
        async with self._lock:
            if not self._running:
                return

            self._running = False
            self._task.cancel()
            await self._task
            await asyncio.gather(*[job.shutdown() for job in self._running_jobs])


class JobContext:
    """
    Context for job, it contains the worker, the room, and the participant.
    """

    def __init__(
        self,
        id: str,
        worker: Worker,
        room: rtc.Room,
        participant: Optional[rtc.RemoteParticipant],
    ) -> None:
        self._id = id
        self._worker = worker
        self._room = room
        self._participant = participant
        self._plugins: list[Plugin] = []
        self._closed = False
        self._lock = asyncio.Lock()
        self._worker._running_jobs.append(self)

    @property
    def id(self) -> str:
        return self._id

    @property
    def room(self) -> rtc.Room:
        return self._room

    @property
    def participant(self) -> Optional[rtc.RemoteParticipant]:
        return self._participant

    def add_plugin(self, plugin: Plugin) -> None:
        self._plugins.append(plugin)

    async def shutdown(self) -> None:
        """
        Shutdown the job and cleanup resources (linked plugins & tasks)
        """
        async with self._lock:
            logging.info(f"shutting down job {self.id}")
            if self._closed:
                return

            # close all plugins
            for p in self._plugins:
                await p.close()

            await self.room.disconnect()

            self._worker._running_jobs.remove(self)
            self._closed = True
            logging.info(f"job {self.id} shutdown")

    async def update_status(
        self,
        status: protocol.agent.JobStatus.ValueType,
        error: str = "",
        user_data: str = "",
    ) -> None:
        await self._worker._send_job_status(self._id, status, error, user_data)


class JobRequest:
    """
    Represents a new job from the server, this worker can either accept or reject it.
    """

    def __init__(
        self,
        worker: Worker,
        job_info: proto_agent.Job,
        simulated: bool = False,
    ) -> None:
        self._worker = worker
        self._info = job_info
        self._room = rtc.Room()
        self._answered = False
        self._simulated = simulated
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def room(self) -> protocol.models.Room:
        return self._info.room

    @property
    def participant(self) -> Optional[protocol.models.ParticipantInfo]:
        if self._info.participant.sid:
            return self._info.participant
        return None

    async def reject(self) -> None:
        """
        Tell the server that we cannot handle the job
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True
            if not self._simulated:
                await self._worker._send_availability(self.id, False)

        logging.info(f"rejected job {self.id}")

    async def accept(
        self,
        agent: Callable[[JobContext], Coroutine],
        should_subscribe: Callable[[
            rtc.TrackPublication, rtc.RemoteParticipant], bool],
        grants: api.VideoGrants = None,
        name: str = "",
        identity: str = "",
        metadata: str = "",
    ) -> None:
        """
        Tell the server that we can handle the job, if the server then assigns the job to us,
        we will connect to the room and call the agent callback
        """
        async with self._lock:
            if self._answered:
                raise Exception("job already answered")

            self._answered = True

            identity = identity or "agent-" + self.id
            grants = grants or api.VideoGrants()
            grants.room_join = True
            grants.agent = True
            grants.room = self.room.name

            jwt = (
                api.AccessToken(self._worker._api_key,
                                self._worker._api_secret)
                .with_identity(identity)
                .with_grants(grants)
                .with_metadata(metadata)
                .with_name(name)
                .to_jwt()
            )

            # raise AssignmentTimeoutError if assignment times out
            if not self._simulated:
                _ = await self._worker._send_availability(self.id, True)

            try:
                should_auto_subscribe = should_subscribe == subscribe_all
                options = rtc.RoomOptions(auto_subscribe=should_auto_subscribe)
                await self._room.connect(self._worker._rtc_url, jwt, options)
            except rtc.ConnectError as e:
                logging.error(
                    f"failed to connect to the room, cancelling job {self.id}: {e}"
                )
                await self._worker._send_job_status(
                    self.id, proto_agent.JobStatus.JS_FAILED, str(e)
                )
                raise e

            sid = self._info.participant.sid
            participant = self._room.participants.get(sid)
            if self.participant is None and sid:
                # cancel the job if the participant cannot be found
                # this can happen if the participant has left the room before the agent gets the job
                logging.warn(
                    f"participant '{sid}' not found, cancelling job {self.id}")
                await self._worker._send_job_status(
                    self.id,
                    proto_agent.JobStatus.JS_FAILED,
                    "participant not found",
                )
                await self._room.disconnect()
                raise JobCancelledError(f"participant '{sid}' not found")

            job_ctx = JobContext(self.id, self._worker,
                                 self._room, participant)
            self._worker._loop.create_task(agent(job_ctx))

            @self._room.on("track_published")
            def on_track_published(publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
                if not should_subscribe(publication, participant):
                    return

                publication.set_subscribed(True)

            for participant in self._room.participants.values():
                for publication in participant.tracks.values():
                    if not should_subscribe(publication):
                        continue

                    publication.set_subscribed(True)

        logging.info(f"accepted job {self.id}")


def _run_worker(
    worker: Worker, loop: Optional[asyncio.AbstractEventLoop] = None,
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
                f"worker started, press Ctrl+C to stop (worker id: {worker.id})"
            )

            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            logging.info(f"shutting down worker {worker._wid}")
            await worker.shutdown()
            logging.info(f"worker {worker._wid} shutdown")

    main_task = loop.create_task(_main_task(worker))
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_task)
    except (GracefulShutdown, KeyboardInterrupt):
        pass
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
    @click.option("--identity", help="particiant identity")
    def simulate_job(room_name: str, identity: str) -> None:
        async def _pre_run() -> Tuple[proto_models.Room, Optional[proto_models.ParticipantInfo]]:
            room_service = api.RoomService(
                worker._rtc_url, worker._api_key, worker._api_secret
            )

            async with aclosing(room_service) as service:
                room = await room_service.create_room(api.CreateRoomRequest(name=room_name))

                participant = None
                if identity:
                    participant = await room_service.get_participant(api.RoomParticipantIdentity(room=room_name, identity=identity))

                return room, participant

        room_info, participant = worker._loop.run_until_complete(_pre_run())
        logging.info(
            f"Simulating job for room {room_info.name} ({room_info.sid}) - Participant: {participant or 'None'}")
        _run_worker(worker, started_cb=lambda _: worker._simulate_job(
            room_info, participant))

    cli()
