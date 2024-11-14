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
import inspect
import math
import multiprocessing as mp
import os
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    TypeVar,
)
from urllib.parse import urljoin, urlparse

import aiohttp
import jwt
from livekit import api, rtc
from livekit.protocol import agent, models

from . import http_server, ipc, utils
from ._exceptions import AssignmentTimeoutError
from .job import (
    JobAcceptArguments,
    JobContext,
    JobExecutorType,
    JobProcess,
    JobRequest,
    RunningJobInfo,
)
from .log import DEV_LEVEL, logger
from .utils.hw import get_cpu_monitor
from .version import __version__

ASSIGNMENT_TIMEOUT = 7.5
UPDATE_LOAD_INTERVAL = 2.5


def _default_initialize_process_fnc(proc: JobProcess) -> Any:
    return


async def _default_request_fnc(ctx: JobRequest) -> None:
    await ctx.accept()


class WorkerType(Enum):
    ROOM = agent.JobType.JT_ROOM
    PUBLISHER = agent.JobType.JT_PUBLISHER


class _DefaultLoadCalc:
    _instance = None

    def __init__(self) -> None:
        self._m_avg = utils.MovingAverage(5)  # avg over 2.5
        self._cpu_monitor = get_cpu_monitor()
        self._thread = threading.Thread(
            target=self._calc_load, daemon=True, name="worker_cpu_load_monitor"
        )
        self._lock = threading.Lock()
        self._thread.start()

    def _calc_load(self) -> None:
        while True:
            cpu_p = self._cpu_monitor.cpu_percent(interval=0.5)
            with self._lock:
                self._m_avg.add_sample(cpu_p)

    def _get_avg(self) -> float:
        with self._lock:
            return self._m_avg.get_avg()

    @classmethod
    def get_load(cls, worker: Worker) -> float:
        if cls._instance is None:
            cls._instance = _DefaultLoadCalc()

        return cls._instance._m_avg.get_avg()


@dataclass
class WorkerPermissions:
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_update_metadata: bool = True
    can_publish_sources: list[models.TrackSource] = field(default_factory=list)
    hidden: bool = False


if sys.platform.startswith("win"):
    # Some python versions on Windows gets a BrokenPipeError when creating a new process
    _default_job_executor_type = JobExecutorType.THREAD
else:
    _default_job_executor_type = JobExecutorType.PROCESS


T = TypeVar("T")


@dataclass(frozen=True)
class _WorkerEnvOption(Generic[T]):
    dev_default: T
    prod_default: T

    @staticmethod
    def getvalue(opt: T | _WorkerEnvOption[T], devmode: bool) -> T:
        if isinstance(opt, _WorkerEnvOption):
            return opt.dev_default if devmode else opt.prod_default
        return opt


# NOTE: this object must be pickle-able
@dataclass
class WorkerOptions:
    entrypoint_fnc: Callable[[JobContext], Awaitable[None]]
    """Entrypoint function that will be called when a job is assigned to this worker."""
    request_fnc: Callable[[JobRequest], Awaitable[None]] = _default_request_fnc
    """Inspect the request and decide if the current worker should handle it.

    When left empty, all jobs are accepted."""
    prewarm_fnc: Callable[[JobProcess], Any] = _default_initialize_process_fnc
    """A function to perform any necessary initialization before the job starts."""
    load_fnc: Callable[[Worker], float] | Callable[[], float] = (
        _DefaultLoadCalc.get_load
    )
    """Called to determine the current load of the worker. Should return a value between 0 and 1."""
    job_executor_type: JobExecutorType = _default_job_executor_type
    """Which executor to use to run jobs. (currently thread or process are supported)"""
    load_threshold: float | _WorkerEnvOption[float] = _WorkerEnvOption(
        dev_default=math.inf, prod_default=0.75
    )
    """When the load exceeds this threshold, the worker will be marked as unavailable.

    Defaults to 0.75 on "production" mode, and is disabled in "development" mode.
    """
    num_idle_processes: int | _WorkerEnvOption[int] = _WorkerEnvOption(
        dev_default=0, prod_default=3
    )
    """Number of idle processes to keep warm."""
    shutdown_process_timeout: float = 60.0
    """Maximum amount of time to wait for a job to shut down gracefully"""
    initialize_process_timeout: float = 10.0
    """Maximum amount of time to wait for a process to initialize/prewarm"""
    permissions: WorkerPermissions = field(default_factory=WorkerPermissions)
    """Permissions that the agent should join the room with."""
    agent_name: str = ""
    """Set agent_name to enable explicit dispatch. When explicit dispatch is enabled, jobs will not be dispatched to rooms automatically. Instead, you can either specify the agent(s) to be dispatched in the end-user's token, or use the AgentDispatch.createDispatch API"""
    worker_type: WorkerType = WorkerType.ROOM
    """Whether to spin up an agent for each room or publisher."""
    max_retry: int = 16
    """Maximum number of times to retry connecting to LiveKit."""
    ws_url: str = "ws://localhost:7880"
    """URL to connect to the LiveKit server.

    By default it uses ``LIVEKIT_URL`` from environment"""
    api_key: str | None = None
    """API key to authenticate with LiveKit.

    By default it uses ``LIVEKIT_API_KEY`` from environment"""
    api_secret: str | None = None
    """API secret to authenticate with LiveKit.

    By default it uses ``LIVEKIT_API_SECRET`` from environment"""
    host: str = ""  # default to all interfaces
    port: int | _WorkerEnvOption[int] = _WorkerEnvOption(
        dev_default=0, prod_default=8081
    )
    """Port for local HTTP server to listen on.

    The HTTP server is used as a health check endpoint.
    """

    def validate_config(self, devmode: bool):
        load_threshold = _WorkerEnvOption.getvalue(self.load_threshold, devmode)
        if load_threshold > 1 and not devmode:
            logger.warning(
                f"load_threshold in prod env must be less than 1, current value: {load_threshold}"
            )


EventTypes = Literal["worker_registered"]


class Worker(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        opts: WorkerOptions,
        *,
        devmode: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        opts.ws_url = opts.ws_url or os.environ.get("LIVEKIT_URL") or ""
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
        self._devmode = devmode

        # using spawn context for all platforms. We may have further optimizations for
        # Linux with forkserver, but for now, this is the safest option
        mp_ctx = mp.get_context("spawn")
        self._proc_pool = ipc.proc_pool.ProcPool(
            initialize_process_fnc=opts.prewarm_fnc,
            job_entrypoint_fnc=opts.entrypoint_fnc,
            num_idle_processes=_WorkerEnvOption.getvalue(
                opts.num_idle_processes, self._devmode
            ),
            loop=self._loop,
            job_executor_type=opts.job_executor_type,
            mp_ctx=mp_ctx,
            initialize_timeout=opts.initialize_process_timeout,
            close_timeout=opts.shutdown_process_timeout,
        )
        self._proc_pool.on("process_started", self._on_process_started)
        self._proc_pool.on("process_closed", self._on_process_closed)
        self._proc_pool.on("process_job_launched", self._on_process_job_launched)

        self._previous_status = agent.WorkerStatus.WS_AVAILABLE

        self._api: api.LiveKitAPI | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._http_server = http_server.HttpServer(
            opts.host,
            _WorkerEnvOption.getvalue(opts.port, self._devmode),
            loop=self._loop,
        )

        self._main_task: asyncio.Task[None] | None = None

    async def run(self):
        if not self._closed:
            raise Exception("worker is already running")

        logger.info(
            "starting worker",
            extra={"version": __version__, "rtc-version": rtc.__version__},
        )

        self._closed = False
        self._proc_pool.start()
        self._api = api.LiveKitAPI(
            self._opts.ws_url, self._opts.api_key, self._opts.api_secret
        )
        self._http_session = aiohttp.ClientSession()
        self._close_future = asyncio.Future(loop=self._loop)

        self._main_task = asyncio.create_task(self._worker_task(), name="worker_task")
        tasks = [
            self._main_task,
            asyncio.create_task(self._http_server.run(), name="http_server"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            if not self._close_future.done():
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
        await self._update_worker_status()

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
        assert self._main_task is not None

        self._closed = True
        self._main_task.cancel()

        await self._proc_pool.aclose()
        await self._http_session.close()
        await self._http_server.aclose()
        await self._api.aclose()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # await asyncio.sleep(0.25)  # see https://github.com/aio-libs/aiohttp/issues/1925
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
        ws: aiohttp.ClientWebSocketResponse | None = None
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
                req.register.type = self._opts.worker_type.value
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
                req.register.agent_name = self._opts.agent_name
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
            finally:
                if ws is not None:
                    await ws.close()

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse):
        closing_ws = False

        async def _load_task():
            """periodically check load and update worker status"""
            interval = utils.aio.interval(UPDATE_LOAD_INTERVAL)
            while True:
                await interval.tick()
                await self._update_worker_status()

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
                elif which == "termination":
                    user_task = self._loop.create_task(
                        self._handle_termination(msg.termination),
                        name="agent_job_termination",
                    )
                    self._tasks.add(user_task)
                    user_task.add_done_callback(self._tasks.discard)

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
        if not self._opts.api_secret:
            raise RuntimeError("api_secret is required to reload jobs")

        for aj in jobs:
            logger.log(
                DEV_LEVEL,
                "reloading job",
                extra={"job_id": aj.job.id, "agent_name": aj.job.agent_name},
            )
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
            extra={
                "id": reg.worker_id,
                "region": reg.server_info.region,
                "protocol": reg.server_info.protocol,
                "node_id": reg.server_info.node_id,
            },
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
            if args.attributes:
                availability_resp.availability.participant_attributes.update(
                    args.attributes
                )
            await self._queue_msg(availability_resp)

            wait_assignment = asyncio.Future[agent.JobAssignment]()
            self._pending_assignments[job_req.id] = wait_assignment

            # the job was accepted by the user, wait for the server assignment
            try:
                await asyncio.wait_for(wait_assignment, ASSIGNMENT_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(
                    f"assignment for job {job_req.id} timed out",
                    extra={"job_request": job_req, "agent_name": self._opts.agent_name},
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
            extra={
                "job_id": msg.job.id,
                "dispatch_id": msg.job.dispatch_id,
                "room_name": msg.job.room.name,
                "agent_name": self._opts.agent_name,
                "resuming": msg.resuming,
            },
        )

        @utils.log_exceptions(logger=logger)
        async def _job_request_task():
            try:
                await self._opts.request_fnc(job_req)
            except Exception:
                logger.exception(
                    "job_request_fnc failed",
                    extra={"job_request": job_req, "agent_name": self._opts.agent_name},
                )

            if not answered:
                logger.warning(
                    "no answer was given inside the job_request_fnc, automatically rejecting the job",
                    extra={"job_request": job_req, "agent_name": self._opts.agent_name},
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
                "received assignment for an unknown job",
                extra={"job": assignment.job, "agent_name": self._opts.agent_name},
            )

    async def _handle_termination(self, msg: agent.JobTermination):
        proc = self._proc_pool.get_by_job_id(msg.job_id)
        if not proc:
            # safe to ignore
            return
        await proc.aclose()

    def _on_process_closed(self, proc: ipc.job_executor.JobExecutor) -> None:
        self._update_job_status_sync(proc)

    def _on_process_started(self, proc: ipc.job_executor.JobExecutor) -> None:
        self._update_job_status_sync(proc)

    def _on_process_job_launched(self, proc: ipc.job_executor.JobExecutor) -> None:
        self._update_job_status_sync(proc)

    async def _update_worker_status(self):
        job_cnt = len(self.active_jobs)
        if self._draining:
            update = agent.UpdateWorkerStatus(
                status=agent.WorkerStatus.WS_FULL, job_count=job_cnt
            )
            msg = agent.WorkerMessage(update_worker=update)
            await self._queue_msg(msg)
            return

        def load_fnc():
            signature = inspect.signature(self._opts.load_fnc)
            parameters = list(signature.parameters.values())
            if len(parameters) == 0:
                return self._opts.load_fnc()  # type: ignore

            return self._opts.load_fnc(self)  # type: ignore

        current_load = await asyncio.get_event_loop().run_in_executor(None, load_fnc)

        is_full = current_load >= _WorkerEnvOption.getvalue(
            self._opts.load_threshold, self._devmode
        )
        currently_available = not is_full and not self._draining

        status = (
            agent.WorkerStatus.WS_AVAILABLE
            if currently_available
            else agent.WorkerStatus.WS_FULL
        )

        update = agent.UpdateWorkerStatus(
            load=current_load, status=status, job_count=job_cnt
        )

        # only log if status has changed
        if self._previous_status != status and not self._draining:
            self._previous_status = status
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

    def _update_job_status_sync(self, proc: ipc.job_executor.JobExecutor) -> None:
        t = self._loop.create_task(self._update_job_status(proc))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _update_job_status(self, proc: ipc.job_executor.JobExecutor) -> None:
        job_info = proc.running_job
        if not job_info:
            return
        status: agent.JobStatus = agent.JobStatus.JS_RUNNING
        if proc.run_status == ipc.job_executor.RunStatus.FINISHED_FAILED:
            status = agent.JobStatus.JS_FAILED
        elif proc.run_status == ipc.job_executor.RunStatus.FINISHED_CLEAN:
            status = agent.JobStatus.JS_SUCCESS
        elif proc.run_status == ipc.job_executor.RunStatus.STARTING:
            status = agent.JobStatus.JS_PENDING

        error: str | None = None
        if proc.exception:
            error = str(proc.exception)
        update = agent.UpdateJobStatus(
            job_id=job_info.job.id, status=status, error=error
        )
        msg = agent.WorkerMessage(update_job=update)
        await self._queue_msg(msg)
