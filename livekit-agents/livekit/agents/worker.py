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
import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Any, Callable, Generic, Literal, TypeVar
from urllib.parse import urljoin, urlparse

import aiohttp
import jwt
from aiohttp import web

from livekit import api, rtc
from livekit.protocol import agent, models

from . import http_server, ipc, utils
from ._exceptions import AssignmentTimeoutError
from .debug import tracing
from .inference_runner import _InferenceRunner
from .job import (
    JobAcceptArguments,
    JobContext,
    JobExecutorType,
    JobProcess,
    JobRequest,
    RunningJobInfo,
)
from .log import DEV_LEVEL, logger
from .plugin import Plugin
from .types import NOT_GIVEN, NotGivenOr
from .utils import is_given
from .utils.hw import get_cpu_monitor
from .version import __version__

ASSIGNMENT_TIMEOUT = 7.5
UPDATE_STATUS_INTERVAL = 2.5
UPDATE_LOAD_INTERVAL = 0.5


def _default_initialize_process_fnc(proc: JobProcess) -> Any:
    return


async def _default_request_fnc(ctx: JobRequest) -> None:
    await ctx.accept()


class WorkerType(Enum):
    ROOM = agent.JobType.JT_ROOM
    PUBLISHER = agent.JobType.JT_PUBLISHER


@dataclass
class SimulateJobInfo:
    room: str
    participant_identity: str | None = None


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
    load_fnc: Callable[[Worker], float] | Callable[[], float] = _DefaultLoadCalc.get_load
    """Called to determine the current load of the worker. Should return a value between 0 and 1."""
    job_executor_type: JobExecutorType = _default_job_executor_type
    """Which executor to use to run jobs. (currently thread or process are supported)"""
    load_threshold: float | _WorkerEnvOption[float] = _WorkerEnvOption(
        dev_default=math.inf, prod_default=0.75
    )
    """When the load exceeds this threshold, the worker will be marked as unavailable.

    Defaults to 0.75 on "production" mode, and is disabled in "development" mode.
    """

    job_memory_warn_mb: float = 500
    """Memory warning threshold in MB. If the job process exceeds this limit, a warning will be logged."""  # noqa: E501
    job_memory_limit_mb: float = 0
    """Maximum memory usage for a job in MB, the job process will be killed if it exceeds this limit.
    Defaults to 0 (disabled).
    """  # noqa: E501

    drain_timeout: int = 1800
    """Number of seconds to wait for current jobs to finish upon receiving TERM or INT signal."""
    num_idle_processes: int | _WorkerEnvOption[int] = _WorkerEnvOption(
        dev_default=0, prod_default=math.ceil(get_cpu_monitor().cpu_count())
    )
    """Number of idle processes to keep warm."""
    shutdown_process_timeout: float = 60.0
    """Maximum amount of time to wait for a job to shut down gracefully"""
    initialize_process_timeout: float = 10.0
    """Maximum amount of time to wait for a process to initialize/prewarm"""
    permissions: WorkerPermissions = field(default_factory=WorkerPermissions)
    """Permissions that the agent should join the room with."""
    agent_name: str = ""
    """Set agent_name to enable explicit dispatch. When explicit dispatch is enabled, jobs will not be dispatched to rooms automatically. Instead, you can either specify the agent(s) to be dispatched in the end-user's token, or use the AgentDispatch.createDispatch API"""  # noqa: E501
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

    _worker_token: str | None = None
    """Internal token."""

    host: str = ""  # default to all interfaces
    port: int | _WorkerEnvOption[int] = _WorkerEnvOption(dev_default=0, prod_default=8081)
    """Port for local HTTP server to listen on.

    The HTTP server is used as a health check endpoint.
    """

    http_proxy: NotGivenOr[str | None] = NOT_GIVEN
    """HTTP proxy used to connect to the LiveKit server.

    By default it uses ``HTTP_PROXY`` or ``HTTPS_PROXY`` from environment
    """
    multiprocessing_context: Literal["spawn", "forkserver"] = (
        "spawn" if not sys.platform.startswith("linux") else "forkserver"
    )
    """The multiprocessing context to use.

    By default it uses "spawn" on all platforms, but "forkserver" on Linux.
    """

    def validate_config(self, devmode: bool) -> None:
        load_threshold = _WorkerEnvOption.getvalue(self.load_threshold, devmode)
        if load_threshold > 1 and not devmode:
            logger.warning(
                f"load_threshold in prod env must be less than 1, current value: {load_threshold}"
            )


@dataclass
class WorkerInfo:
    http_port: int


EventTypes = Literal["worker_started", "worker_registered"]


class Worker(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        opts: WorkerOptions,
        *,
        devmode: bool = True,
        register: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        opts.ws_url = opts.ws_url or os.environ.get("LIVEKIT_URL") or ""
        opts.api_key = opts.api_key or os.environ.get("LIVEKIT_API_KEY") or ""
        opts.api_secret = opts.api_secret or os.environ.get("LIVEKIT_API_SECRET") or ""
        opts._worker_token = os.environ.get("LIVEKIT_WORKER_TOKEN") or None

        if not opts.ws_url:
            raise ValueError("ws_url is required, or add LIVEKIT_URL in your environment")

        if not opts.api_key:
            raise ValueError("api_key is required, or add LIVEKIT_API_KEY in your environment")

        if not opts.api_secret:
            raise ValueError(
                "api_secret is required, or add LIVEKIT_API_SECRET in your environment"
            )

        if opts.job_memory_limit_mb > 0 and opts.job_executor_type != JobExecutorType.PROCESS:
            logger.warning(
                "max_job_memory_usage is only supported for process-based job executors, "
                "ignoring max_job_memory_usage"
            )

        if not is_given(opts.http_proxy):
            opts.http_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        self._opts = opts
        self._loop = loop or asyncio.get_event_loop()

        self._id = "unregistered"
        self._closed, self._draining, self._connecting = True, False, False
        self._tasks = set[asyncio.Task[Any]]()
        self._pending_assignments: dict[str, asyncio.Future[agent.JobAssignment]] = {}
        self._close_future: asyncio.Future[None] | None = None
        self._msg_chan = utils.aio.Chan[agent.WorkerMessage](128, loop=self._loop)
        self._devmode = devmode
        self._register = register

        self._mp_ctx = mp.get_context(self._opts.multiprocessing_context)

        self._inference_executor: ipc.inference_proc_executor.InferenceProcExecutor | None = None
        if len(_InferenceRunner.registered_runners) > 0:
            self._inference_executor = ipc.inference_proc_executor.InferenceProcExecutor(
                runners=_InferenceRunner.registered_runners,
                initialize_timeout=30,
                close_timeout=5,
                memory_warn_mb=2000,
                memory_limit_mb=0,  # no limit
                ping_interval=5,
                ping_timeout=60,
                high_ping_threshold=2.5,
                mp_ctx=self._mp_ctx,
                loop=self._loop,
                http_proxy=opts.http_proxy or None,
            )

        self._proc_pool = ipc.proc_pool.ProcPool(
            initialize_process_fnc=opts.prewarm_fnc,
            job_entrypoint_fnc=opts.entrypoint_fnc,
            num_idle_processes=_WorkerEnvOption.getvalue(opts.num_idle_processes, self._devmode),
            loop=self._loop,
            job_executor_type=opts.job_executor_type,
            inference_executor=self._inference_executor,
            mp_ctx=self._mp_ctx,
            initialize_timeout=opts.initialize_process_timeout,
            close_timeout=opts.shutdown_process_timeout,
            memory_warn_mb=opts.job_memory_warn_mb,
            memory_limit_mb=opts.job_memory_limit_mb,
            http_proxy=opts.http_proxy or None,
        )

        self._previous_status = agent.WorkerStatus.WS_AVAILABLE

        self._api: api.LiveKitAPI | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._http_server = http_server.HttpServer(
            opts.host,
            _WorkerEnvOption.getvalue(opts.port, self._devmode),
            loop=self._loop,
        )

        async def health_check(_: Any) -> web.Response:
            return web.Response(text="OK")

        async def worker(_: Any) -> web.Response:
            body = json.dumps(
                {
                    "agent_name": self._opts.agent_name,
                    "worker_type": agent.JobType.Name(self._opts.worker_type.value),
                    "active_jobs": len(self.active_jobs),
                }
            )
            return web.Response(body=body, content_type="application/json")

        self._http_server.app.add_routes([web.get("/", health_check)])
        self._http_server.app.add_routes([web.get("/worker", worker)])
        self._http_server.app.add_subapp("/debug", tracing._create_tracing_app(self))

        self._conn_task: asyncio.Task[None] | None = None
        self._load_task: asyncio.Task[None] | None = None

        self._worker_load: float = 0.0
        self._worker_load_graph = tracing.Tracing.add_graph(
            title="worker_load",
            x_label="time",
            y_label="load",
            x_type="time",
            y_range=(0, 1),
            max_data_points=int(1 / UPDATE_LOAD_INTERVAL * 30),
        )

        default_num_idle_processes = _WorkerEnvOption.getvalue(
            self._opts.num_idle_processes, self._devmode
        )
        self._num_idle_target_graph = tracing.Tracing.add_graph(
            title="num_idle_processes_target",
            x_label="time",
            y_label="target",
            x_type="time",
            y_range=(0, default_num_idle_processes),
            max_data_points=int(1 / UPDATE_LOAD_INTERVAL * 30),
        )

        self._num_idle_process_graph = tracing.Tracing.add_graph(
            title="num_idle_processes",
            x_label="time",
            y_label="idle",
            x_type="time",
            y_range=(0, default_num_idle_processes),
            max_data_points=int(1 / UPDATE_LOAD_INTERVAL * 30),
        )

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(http_port=self._http_server.port)

    async def run(self) -> None:
        if not self._closed:
            raise Exception("worker is already running")

        logger.info(
            "starting worker",
            extra={"version": __version__, "rtc-version": rtc.__version__},
        )

        if self._opts.multiprocessing_context == "forkserver":
            plugin_packages = [p.package for p in Plugin.registered_plugins]
            logger.info("preloading plugins", extra={"packages": plugin_packages})
            self._mp_ctx.set_forkserver_preload(plugin_packages)

        if self._inference_executor is not None:
            logger.info("starting inference executor")
            await self._inference_executor.start()
            await self._inference_executor.initialize()

        self._closed = False

        def _update_job_status(proc: ipc.job_executor.JobExecutor) -> None:
            t = self._loop.create_task(self._update_job_status(proc))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

        await self._http_server.start()

        self._proc_pool.on("process_started", _update_job_status)
        self._proc_pool.on("process_closed", _update_job_status)
        self._proc_pool.on("process_job_launched", _update_job_status)
        await self._proc_pool.start()

        self._http_session = aiohttp.ClientSession(proxy=self._opts.http_proxy or None)
        self._api = api.LiveKitAPI(
            self._opts.ws_url, self._opts.api_key, self._opts.api_secret, session=self._http_session
        )
        self._close_future = asyncio.Future(loop=self._loop)

        @utils.log_exceptions(logger=logger)
        async def _load_task() -> None:
            """periodically check load"""
            interval = utils.aio.interval(UPDATE_LOAD_INTERVAL)
            while True:
                await interval.tick()

                def load_fnc() -> float:
                    signature = inspect.signature(self._opts.load_fnc)
                    parameters = list(signature.parameters.values())
                    if len(parameters) == 0:
                        return self._opts.load_fnc()  # type: ignore

                    return self._opts.load_fnc(self)  # type: ignore

                self._worker_load = await asyncio.get_event_loop().run_in_executor(None, load_fnc)

                load_threshold = _WorkerEnvOption.getvalue(self._opts.load_threshold, self._devmode)
                default_num_idle_processes = _WorkerEnvOption.getvalue(
                    self._opts.num_idle_processes, self._devmode
                )

                if not math.isinf(load_threshold):
                    active_jobs = len(self.active_jobs)
                    if active_jobs > 0:
                        job_load = self._worker_load / len(self.active_jobs)
                        if job_load > 0.0:
                            available_load = max(load_threshold - self._worker_load, 0.0)
                            available_job = min(
                                math.ceil(available_load / job_load), default_num_idle_processes
                            )
                            self._proc_pool.set_target_idle_processes(available_job)
                    else:
                        self._proc_pool.set_target_idle_processes(default_num_idle_processes)

                self._num_idle_target_graph.plot(time.time(), self._proc_pool.target_idle_processes)
                self._num_idle_process_graph.plot(
                    time.time(), self._proc_pool._warmed_proc_queue.qsize()
                )
                self._worker_load_graph.plot(time.time(), self._worker_load)

        tasks = []
        self._load_task = asyncio.create_task(_load_task(), name="load_task")
        tasks.append(self._load_task)

        if self._register:
            self._conn_task = asyncio.create_task(self._connection_task(), name="worker_conn_task")
            tasks.append(self._conn_task)

        self.emit("worker_started")

        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            if not self._close_future.done():
                self._close_future.set_result(None)

    @property
    def id(self) -> str:
        return self._id

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return [proc.running_job for proc in self._proc_pool.processes if proc.running_job]

    async def drain(self, timeout: int | None = None) -> None:
        """When timeout isn't None, it will raise asyncio.TimeoutError if the processes didn't finish in time."""  # noqa: E501
        if self._draining:
            return

        logger.info("draining worker", extra={"id": self.id, "timeout": timeout})
        self._draining = True
        await self._update_worker_status()

        async def _join_jobs() -> None:
            for proc in self._proc_pool.processes:
                if proc.running_job:
                    await proc.join()

        if timeout:
            await asyncio.wait_for(_join_jobs(), timeout)  # raises asyncio.TimeoutError on timeout
        else:
            await _join_jobs()

    async def simulate_job(
        self,
        info: SimulateJobInfo | str,
    ) -> None:
        """
        Simulate a job by creating a room and participant.

        Args:
            info: SimulateJobInfo or a join token for an existing room
        """
        assert self._api is not None
        # TODO(theomonnom): some fake information can still be found in the token

        from livekit.protocol.models import Room

        room = info.room if isinstance(info, SimulateJobInfo) else "unknown-room"
        participant_identity = (
            info.participant_identity
            if isinstance(info, SimulateJobInfo)
            else "unknown-participant"
        )
        agent_id = utils.shortuuid("simulated-agent-")

        room_info = Room(sid=utils.shortuuid("RM_"), name=room)
        participant_info = None

        if isinstance(info, SimulateJobInfo):
            from .cli import cli

            if cli.CLI_ARGUMENTS is None or not cli.CLI_ARGUMENTS.console:
                room_info = await self._api.room.create_room(api.CreateRoomRequest(name=room))
                if participant_identity:
                    participant_info = await self._api.room.get_participant(
                        api.RoomParticipantIdentity(room=room, identity=participant_identity)
                    )

            token = (
                api.AccessToken(self._opts.api_key, self._opts.api_secret)
                .with_identity(agent_id)
                .with_kind("agent")
                .with_grants(api.VideoGrants(room_join=True, room=room, agent=True))
                .to_jwt()
            )
        else:
            token = info

        job = agent.Job(
            id=utils.shortuuid("simulated-job-"),
            room=room_info,
            type=agent.JobType.JT_ROOM,
            participant=participant_info,
        )

        running_info = RunningJobInfo(
            worker_id=self._id,
            accept_arguments=JobAcceptArguments(identity=agent_id, name="", metadata=""),
            job=job,
            url=self._opts.ws_url,
            token=token,
        )

        await self._proc_pool.launch_job(running_info)

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

        if self._conn_task is not None:
            await utils.aio.cancel_and_wait(self._conn_task)

        if self._load_task is not None:
            await utils.aio.cancel_and_wait(self._load_task)

        await self._proc_pool.aclose()

        if self._inference_executor is not None:
            await self._inference_executor.aclose()

        await self._http_session.close()
        await self._http_server.aclose()
        await self._api.aclose()  # type: ignore

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

    @utils.log_exceptions(logger=logger)
    async def _connection_task(self) -> None:
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

                params = {}
                if self._opts._worker_token:
                    params["worker_token"] = self._opts._worker_token

                ws = await self._http_session.ws_connect(
                    agent_url,
                    headers=headers,
                    params=params,
                    autoping=True,
                    proxy=self._opts.http_proxy or None,
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
                    ) from None

                retry_delay = min(retry_count * 2, 10)
                retry_count += 1

                logger.warning(
                    f"failed to connect to livekit, retrying in {retry_delay}s", exc_info=e
                )
                await asyncio.sleep(retry_delay)
            finally:
                if ws is not None:
                    await ws.close()

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        closing_ws = False

        async def _load_task() -> None:
            """periodically update worker status"""
            interval = utils.aio.interval(UPDATE_STATUS_INTERVAL)
            while True:
                await interval.tick()
                await self._update_worker_status()

        async def _send_task() -> None:
            nonlocal closing_ws
            while True:
                try:
                    msg = await self._msg_chan.recv()
                    await ws.send_bytes(msg.SerializeToString())
                except utils.aio.ChanClosed:
                    closing_ws = True
                    return

        async def _recv_task() -> None:
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
                server_msg = agent.ServerMessage()
                server_msg.ParseFromString(data)
                which = server_msg.WhichOneof("message")
                if which == "availability":
                    self._handle_availability(server_msg.availability)
                elif which == "assignment":
                    self._handle_assignment(server_msg.assignment)
                elif which == "termination":
                    user_task = self._loop.create_task(
                        self._handle_termination(server_msg.termination),
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
            await utils.aio.cancel_and_wait(*tasks)

    async def _reload_jobs(self, jobs: list[RunningJobInfo]) -> None:
        if not self._opts.api_secret:
            raise RuntimeError("api_secret is required to reload jobs")

        for aj in jobs:
            logger.log(
                DEV_LEVEL,
                "reloading job",
                extra={"job_id": aj.job.id, "agent_name": aj.job.agent_name},
            )

            # take the original jwt token and extend it while keeping all the same data that was generated  # noqa: E501
            # by the SFU for the original join token.
            original_token = aj.token
            decoded = jwt.decode(original_token, self._opts.api_secret, algorithms=["HS256"])
            decoded["exp"] = int(datetime.datetime.now(datetime.timezone.utc).timestamp()) + 3600
            running_info = RunningJobInfo(
                accept_arguments=aj.accept_arguments,
                job=aj.job,
                url=self._opts.ws_url,
                token=jwt.encode(decoded, self._opts.api_secret, algorithm="HS256"),
                worker_id=aj.worker_id,
            )
            await self._proc_pool.launch_job(running_info)

    def _handle_register(self, reg: agent.RegisterWorkerResponse) -> None:
        self._id = reg.worker_id
        logger.info(
            "registered worker",
            extra={
                "id": reg.worker_id,
                "url": self._opts.ws_url,
                "region": reg.server_info.region,
                "protocol": reg.server_info.protocol,
            },
        )
        self.emit("worker_registered", reg.worker_id, reg.server_info)

    def _handle_availability(self, msg: agent.AvailabilityRequest) -> None:
        task = self._loop.create_task(self._answer_availability(msg))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _answer_availability(self, msg: agent.AvailabilityRequest) -> None:
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
                availability_resp.availability.participant_attributes.update(args.attributes)
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
                raise AssignmentTimeoutError() from None

            job_assign = wait_assignment.result()
            running_info = RunningJobInfo(
                accept_arguments=args,
                job=msg.job,
                url=job_assign.url or self._opts.ws_url,
                token=job_assign.token,
                worker_id=self._id,
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
        async def _job_request_task() -> None:
            try:
                await self._opts.request_fnc(job_req)
            except Exception:
                logger.exception(
                    "job_request_fnc failed",
                    extra={"job_request": job_req, "agent_name": self._opts.agent_name},
                )

            if not answered:
                logger.warning(
                    "no answer was given inside the job_request_fnc, automatically rejecting the job",  # noqa: E501
                    extra={"job_request": job_req, "agent_name": self._opts.agent_name},
                )
                await _on_reject()

        user_task = self._loop.create_task(_job_request_task(), name="job_request")
        self._tasks.add(user_task)
        user_task.add_done_callback(self._tasks.discard)

    def _handle_assignment(self, assignment: agent.JobAssignment) -> None:
        if assignment.job.id in self._pending_assignments:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut = self._pending_assignments.pop(assignment.job.id)
                fut.set_result(assignment)
        else:
            logger.warning(
                "received assignment for an unknown job",
                extra={"job": assignment.job, "agent_name": self._opts.agent_name},
            )

    async def _handle_termination(self, msg: agent.JobTermination) -> None:
        proc = self._proc_pool.get_by_job_id(msg.job_id)
        if not proc:
            # safe to ignore
            return
        await proc.aclose()

    async def _update_worker_status(self) -> None:
        job_cnt = len(self.active_jobs)
        if self._draining:
            update = agent.UpdateWorkerStatus(status=agent.WorkerStatus.WS_FULL, job_count=job_cnt)
            msg = agent.WorkerMessage(update_worker=update)
            await self._queue_msg(msg)
            return

        load_threshold = _WorkerEnvOption.getvalue(self._opts.load_threshold, self._devmode)
        is_full = self._worker_load >= load_threshold
        currently_available = not is_full and not self._draining

        status = (
            agent.WorkerStatus.WS_AVAILABLE if currently_available else agent.WorkerStatus.WS_FULL
        )

        update = agent.UpdateWorkerStatus(load=self._worker_load, status=status, job_count=job_cnt)

        # only log if status has changed
        if self._previous_status != status and not self._draining:
            self._previous_status = status
            extra = {
                "load": self._worker_load,
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

    async def _update_job_status(self, proc: ipc.job_executor.JobExecutor) -> None:
        job_info = proc.running_job
        if job_info is None:
            return

        status: agent.JobStatus = agent.JobStatus.JS_RUNNING
        if proc.status == ipc.job_executor.JobStatus.FAILED:
            status = agent.JobStatus.JS_FAILED
        elif proc.status == ipc.job_executor.JobStatus.SUCCESS:
            status = agent.JobStatus.JS_SUCCESS
        elif proc.status == ipc.job_executor.JobStatus.RUNNING:
            status = agent.JobStatus.JS_RUNNING

        update = agent.UpdateJobStatus(job_id=job_info.job.id, status=status, error="")
        msg = agent.WorkerMessage(update_job=update)
        await self._queue_msg(msg)
