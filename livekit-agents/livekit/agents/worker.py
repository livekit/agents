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
import contextlib
import datetime
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Coroutine, Literal
from urllib.parse import urljoin, urlparse

import aiohttp
import jwt
import psutil
from livekit import api
from livekit.protocol import agent, models

from . import http_server, ipc, utils
from .exceptions import AssignmentTimeoutError
from .job import JobAcceptArguments, JobContext, JobProcess, JobRequest, RunningJobInfo
from .log import DEV_LEVEL, logger
from .version import __version__

MAX_RECONNECT_ATTEMPTS = 3
ASSIGNMENT_TIMEOUT = 7.5
UPDATE_LOAD_INTERVAL = 10.0


def _default_initialize_process_fnc(proc: JobProcess) -> Any:
    return


async def _default_request_fnc(ctx: JobRequest) -> None:
    await ctx.accept()


def _default_cpu_load_fnc() -> float:
    percent = psutil.cpu_percent(1.0)
    return percent / 100


@dataclass
class WorkerPermissions:
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_update_metadata: bool = True
    can_publish_sources: list[models.TrackSource] = field(default_factory=list)
    hidden: bool = False


# NOTE: this object must be pickle-able
@dataclass
class WorkerOptions:
    entrypoint_fnc: Callable[[JobContext], Coroutine]
    request_fnc: Callable[[JobRequest], Coroutine] = _default_request_fnc
    prewarm_fnc: Callable[[JobProcess], Any] = _default_initialize_process_fnc
    load_fnc: Callable[[], float] = _default_cpu_load_fnc
    load_threshold: float = 0.65
    num_idle_processes: int = 3
    shutdown_process_timeout: float = 60.0
    initialize_process_timeout: float = 10.0
    permissions: WorkerPermissions = field(default_factory=WorkerPermissions)
    worker_type: agent.JobType = agent.JobType.JT_ROOM
    max_retry: int = MAX_RECONNECT_ATTEMPTS
    ws_url: str = "ws://localhost:7880"
    api_key: str | None = None
    api_secret: str | None = None
    host: str = ""  # default to all interfaces
    port: int = 8081


EventTypes = Literal["worker_registered"]


class Worker(utils.EventEmitter[EventTypes]):
    def __init__(
        self, opts: WorkerOptions, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        super().__init__()
        opts.ws_url = opts.ws_url or opts.ws_url or os.environ.get("LIVEKIT_URL") or ""
        opts.api_key = opts.api_key or os.environ.get("LIVEKIT_API_KEY") or ""
        opts.api_secret = opts.api_secret or os.environ.get("LIVEKIT_API_SECRET") or ""

        if not opts.ws_url:
            raise ValueError(
                "ws_url is required, or add LIVEKIT_URL in your environment"
            )

        if not opts.api_key:
            raise ValueError(
                "api_key is required, or add LIVEKIT_API_KEY in your environment"
            )

        if not opts.api_secret:
            raise ValueError(
                "api_secret is required, or add LIVEKIT_API_SECRET in your environment"
            )

        self._opts = opts
        self._loop = loop or asyncio.get_event_loop()

        self._id = "unregistered"
        self._closed, self._draining, self._connecting = True, False, False
        self._tasks = set[asyncio.Task[Any]]()
        self._pending_assignments: dict[str, asyncio.Future[agent.JobAssignment]] = {}
        self._close_future: asyncio.Future[None] | None = None
        self._msg_chan = utils.aio.Chan[agent.WorkerMessage](128, loop=self._loop)

        # using spawn context for all platforms. We may have further optimizations for
        # Linux with forkserver, but for now, this is the safest option
        mp_ctx = mp.get_context("spawn")
        self._proc_pool = ipc.proc_pool.ProcPool(
            initialize_process_fnc=opts.prewarm_fnc,
            job_entrypoint_fnc=opts.entrypoint_fnc,
            num_idle_processes=opts.num_idle_processes,
            loop=self._loop,
            mp_ctx=mp_ctx,
            initialize_timeout=opts.initialize_process_timeout,
            close_timeout=opts.shutdown_process_timeout,
        )

        self._api: api.LiveKitAPI | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._http_server = http_server.HttpServer(
            opts.host, opts.port, loop=self._loop
        )

    async def run(self):
        if not self._closed:
            raise Exception("worker is already running")

        logger.info("starting worker", extra={"version": __version__})

        self._closed = False
        self._proc_pool.start()
        self._api = api.LiveKitAPI(
            self._opts.ws_url, self._opts.api_key, self._opts.api_secret
        )
        self._http_session = aiohttp.ClientSession()
        self._close_future = asyncio.Future(loop=self._loop)

        try:
            await asyncio.gather(self._worker_task(), self._http_server.run())
        finally:
            self._close_future.set_result(None)

    @property
    def id(self) -> str:
        return self._id

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return [
            proc.running_job for proc in self._proc_pool.processes if proc.running_job
        ]

    async def drain(self, timeout: int | None = None) -> None:
        """When timeout isn't None, it will raise asyncio.TimeoutError if the processes didn't finish in time."""
        if self._draining:
            return

        logger.info("draining worker", extra={"id": self.id, "timeout": timeout})
        self._draining = True

        # exit the queue
        update_worker = agent.WorkerMessage(
            update_worker=agent.UpdateWorkerStatus(status=agent.WorkerStatus.WS_FULL)
        )
        await self._queue_msg(update_worker)

        async def _join_jobs():
            for proc in self._proc_pool.processes:
                if proc.running_job:
                    await proc.join()

        if timeout:
            await asyncio.wait_for(
                _join_jobs(), timeout
            )  # raises asyncio.TimeoutError on timeout
        else:
            await _join_jobs()

    async def simulate_job(
        self, room: str, participant_identity: str | None = None
    ) -> None:
        assert self._api is not None

        room_obj = await self._api.room.create_room(api.CreateRoomRequest(name=room))
        participant = None
        if participant_identity:
            participant = await self._api.room.get_participant(
                api.RoomParticipantIdentity(room=room, identity=participant_identity)
            )

        msg = agent.WorkerMessage()
        msg.simulate_job.room.CopyFrom(room_obj)
        if participant:
            msg.simulate_job.participant.CopyFrom(participant)

        await self._queue_msg(msg)

    async def aclose(self) -> None:
        if self._closed:
            if self._close_future is not None:
                await self._close_future
            return

        logger.info("shutting down worker", extra={"id": self.id})

        assert self._close_future is not None
        assert self._http_session is not None
        assert self._api is not None

        self._closed = True

        await self._proc_pool.aclose()
        await self._http_session.close()
        await self._http_server.aclose()
        await self._api.aclose()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await asyncio.sleep(0.25)  # see https://github.com/aio-libs/aiohttp/issues/1925
        self._msg_chan.close()
        await self._close_future

    async def _queue_msg(self, msg: agent.WorkerMessage) -> None:
        """_queue_msg raises aio.ChanClosed when the worker is closing/closed"""
        if self._connecting:
            which = msg.WhichOneof("message")
            if which == "update_worker":
                return
            elif which == "ping":
                return

        await self._msg_chan.send(msg)

    async def _worker_task(self) -> None:
        assert self._http_session is not None

        retry_count = 0
        while not self._closed:
            try:
                self._connecting = True
                join_jwt = (
                    api.AccessToken(self._opts.api_key, self._opts.api_secret)
                    .with_grants(api.VideoGrants(agent=True))
                    .to_jwt()
                )

                headers = {"Authorization": f"Bearer {join_jwt}"}

                parse = urlparse(self._opts.ws_url)
                scheme = parse.scheme
                if scheme.startswith("http"):
                    scheme = scheme.replace("http", "ws")

                path_parts = [f"{scheme}://{parse.netloc}", parse.path, "/agent"]
                agent_url = reduce(urljoin, path_parts)

                ws = await self._http_session.ws_connect(
                    agent_url, headers=headers, autoping=True
                )

                retry_count = 0

                # register the worker
                req = agent.WorkerMessage()
                req.register.type = self._opts.worker_type
                req.register.allowed_permissions.CopyFrom(
                    models.ParticipantPermission(
                        can_publish=self._opts.permissions.can_publish,
                        can_subscribe=self._opts.permissions.can_subscribe,
                        can_publish_data=self._opts.permissions.can_publish_data,
                        can_update_metadata=self._opts.permissions.can_update_metadata,
                        can_publish_sources=self._opts.permissions.can_publish_sources,
                        hidden=self._opts.permissions.hidden,
                        agent=True,
                    )
                )
                req.register.namespace = "default"
                req.register.version = __version__
                await ws.send_bytes(req.SerializeToString())

                # wait for the register response before running this connection
                first_msg_b = await ws.receive_bytes()
                msg = agent.ServerMessage()
                msg.ParseFromString(first_msg_b)

                if not msg.HasField("register"):
                    raise Exception("expected register response as first message")

                self._handle_register(msg.register)
                self._connecting = False

                await self._run_ws(ws)
            except Exception as e:
                if self._closed:
                    break

                if retry_count >= self._opts.max_retry:
                    raise RuntimeError(
                        f"failed to connect to livekit after {retry_count} attempts",
                    )

                retry_delay = min(retry_count * 2, 10)
                retry_count += 1

                logger.warning(
                    f"failed to connect to livekit, retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse):
        closing_ws = False

        async def _load_task():
            """periodically check load and update worker status"""
            interval = utils.aio.interval(UPDATE_LOAD_INTERVAL)
            current_status = agent.WorkerStatus.WS_AVAILABLE
            while True:
                await interval.tick()

                old_status = current_status
                current_load = await asyncio.get_event_loop().run_in_executor(
                    None, self._opts.load_fnc
                )

                is_full = current_load >= self._opts.load_threshold
                currently_available = not is_full and not self._draining

                current_status = (
                    agent.WorkerStatus.WS_AVAILABLE
                    if currently_available
                    else agent.WorkerStatus.WS_FULL
                )

                update = agent.UpdateWorkerStatus(
                    load=current_load, status=current_status
                )

                # only log if status has changed
                if old_status != current_status and not self._draining:
                    extra = {
                        "load": current_load,
                        "threshold": self._opts.load_threshold,
                    }
                    if is_full:
                        logger.info(
                            "worker is at full capacity, marking as unavailable",
                            extra=extra,
                        )
                    else:
                        logger.info(
                            "worker is below capacity, marking as available",
                            extra=extra,
                        )

                msg = agent.WorkerMessage(update_worker=update)
                with contextlib.suppress(utils.aio.ChanClosed):
                    await self._queue_msg(msg)

        async def _send_task():
            nonlocal closing_ws
            while True:
                try:
                    msg = await self._msg_chan.recv()
                    await ws.send_bytes(msg.SerializeToString())
                except utils.aio.ChanClosed:
                    closing_ws = True
                    return

        async def _recv_task():
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return

                    raise Exception("worker connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning("unexpected message type: %s", msg.type)
                    continue

                data = msg.data
                msg = agent.ServerMessage()
                msg.ParseFromString(data)
                which = msg.WhichOneof("message")
                if which == "availability":
                    self._handle_availability(msg.availability)
                elif which == "assignment":
                    self._handle_assignment(msg.assignment)

        tasks = [
            asyncio.create_task(_load_task()),
            asyncio.create_task(_send_task()),
            asyncio.create_task(_recv_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _reload_jobs(self, jobs: list[RunningJobInfo]) -> None:
        for aj in jobs:
            logger.log(DEV_LEVEL, "reloading job", extra={"job_id": aj.job.id})
            url = self._opts.ws_url

            # take the original jwt token and extend it while keeping all the same data that was generated
            # by the SFU for the original join token.
            original_token = aj.token
            decoded = jwt.decode(
                original_token, self._opts.api_secret, algorithms=["HS256"]
            )
            decoded["exp"] = (
                int(datetime.datetime.now(datetime.timezone.utc).timestamp()) + 3600
            )
            running_info = RunningJobInfo(
                accept_arguments=aj.accept_arguments,
                job=aj.job,
                url=url,
                token=jwt.encode(decoded, self._opts.api_secret, algorithm="HS256"),
            )
            await self._proc_pool.launch_job(running_info)

    def _handle_register(self, reg: agent.RegisterWorkerResponse):
        self._id = reg.worker_id
        logger.info(
            "registered worker",
            extra={"id": reg.worker_id, "server_info": reg.server_info},
        )
        self.emit("worker_registered", reg.worker_id, reg.server_info)

    def _handle_availability(self, msg: agent.AvailabilityRequest):
        task = self._loop.create_task(self._answer_availability(msg))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _answer_availability(self, msg: agent.AvailabilityRequest):
        """Ask the user if they want to accept this job and forward the answer to the server.
        If we get the job assigned, we start a new process."""

        answered = False

        async def _on_reject() -> None:
            nonlocal answered
            answered = True

            availability_resp = agent.WorkerMessage()
            availability_resp.availability.job_id = msg.job.id
            availability_resp.availability.available = False
            await self._queue_msg(availability_resp)

        async def _on_accept(args: JobAcceptArguments) -> None:
            nonlocal answered
            answered = True

            availability_resp = agent.WorkerMessage()
            availability_resp.availability.job_id = msg.job.id
            availability_resp.availability.available = True
            availability_resp.availability.participant_identity = args.identity
            availability_resp.availability.participant_name = args.name
            availability_resp.availability.participant_metadata = args.metadata
            await self._queue_msg(availability_resp)

            wait_assignment = asyncio.Future[agent.JobAssignment]()
            self._pending_assignments[job_req.id] = wait_assignment

            # the job was accepted by the user, wait for the server assignment
            try:
                await asyncio.wait_for(wait_assignment, ASSIGNMENT_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(
                    f"assignment for job {job_req.id} timed out",
                    extra={"job_request": job_req},
                )
                raise AssignmentTimeoutError()

            job_assign = wait_assignment.result()
            running_info = RunningJobInfo(
                accept_arguments=args,
                job=msg.job,
                url=job_assign.url or self._opts.ws_url,
                token=job_assign.token,
            )

            await self._proc_pool.launch_job(running_info)

        job_req = JobRequest(job=msg.job, on_reject=_on_reject, on_accept=_on_accept)

        logger.info(
            "received job request",
            extra={"job_request": msg.job, "resuming": msg.resuming},
        )

        @utils.log_exceptions(logger=logger)
        async def _job_request_task():
            try:
                await self._opts.request_fnc(job_req)
            except Exception:
                logger.exception(
                    "job_request_fnc failed", extra={"job_request": job_req}
                )

            if not answered:
                logger.warning(
                    "no answer was given inside the job_request_fnc, automatically rejecting the job",
                    extra={"job_request": job_req},
                )
                await _on_reject()

        user_task = self._loop.create_task(_job_request_task(), name="job_request")
        self._tasks.add(user_task)
        user_task.add_done_callback(self._tasks.discard)

    def _handle_assignment(self, assignment: agent.JobAssignment):
        if assignment.job.id in self._pending_assignments:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut = self._pending_assignments.pop(assignment.job.id)
                fut.set_result(assignment)
        else:
            logger.warning(
                "received assignment for an unknown job", extra={"job": assignment.job}
            )
