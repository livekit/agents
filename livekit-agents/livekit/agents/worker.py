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
from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, Literal, TypeVar, overload
from urllib.parse import urljoin, urlparse

import aiohttp
import jwt
from aiohttp import web
from google.protobuf.json_format import MessageToDict

from livekit import api, rtc
from livekit.protocol import agent, models

from . import ipc, telemetry, utils
from ._exceptions import AssignmentTimeoutError
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
from .utils import http_server, is_given
from .utils.hw import get_cpu_monitor
from .version import __version__

ASSIGNMENT_TIMEOUT = 7.5
UPDATE_STATUS_INTERVAL = 2.5
UPDATE_LOAD_INTERVAL = 0.5
HEARTBEAT_INTERVAL = 30


def _default_setup_fnc(proc: JobProcess) -> Any:
    return


async def _default_request_fnc(ctx: JobRequest) -> None:
    await ctx.accept()


class ServerType(Enum):
    ROOM = agent.JobType.JT_ROOM
    PUBLISHER = agent.JobType.JT_PUBLISHER


WorkerType = ServerType


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
    def get_load(cls, worker: AgentServer) -> float:
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
class ServerEnvOption(Generic[T]):
    dev_default: T
    prod_default: T

    @staticmethod
    def getvalue(opt: T | ServerEnvOption[T], devmode: bool) -> T:
        if isinstance(opt, ServerEnvOption):
            return opt.dev_default if devmode else opt.prod_default
        return opt


_default_load_threshold = ServerEnvOption(dev_default=math.inf, prod_default=0.7)
_default_permissions = WorkerPermissions()


# NOTE: this object must be pickle-able
@dataclass
class ServerOptions:
    entrypoint_fnc: Callable[[JobContext], Awaitable[None]]
    """Entrypoint function that will be called when a job is assigned to this worker."""
    request_fnc: Callable[[JobRequest], Awaitable[None]] = _default_request_fnc
    """Inspect the request and decide if the current worker should handle it.

    When left empty, all jobs are accepted."""
    prewarm_fnc: Callable[[JobProcess], Any] = _default_setup_fnc
    """A function to perform any necessary initialization before the job starts."""
    load_fnc: Callable[[AgentServer], float] | Callable[[], float] = _DefaultLoadCalc.get_load
    """Called to determine the current load of the worker. Should return a value between 0 and 1."""
    job_executor_type: JobExecutorType = _default_job_executor_type
    """Which executor to use to run jobs. (currently thread or process are supported)"""
    load_threshold: float | ServerEnvOption[float] = _default_load_threshold
    """When the load exceeds this threshold, the worker will be marked as unavailable.

    Defaults to 0.7 on "production" mode, and is disabled in "development" mode.
    """

    job_memory_warn_mb: float = 500
    """Memory warning threshold in MB. If the job process exceeds this limit, a warning will be logged."""  # noqa: E501
    job_memory_limit_mb: float = 0
    """Maximum memory usage for a job in MB, the job process will be killed if it exceeds this limit.
    Defaults to 0 (disabled).
    """  # noqa: E501

    drain_timeout: int = 1800
    """Number of seconds to wait for current jobs to finish upon receiving TERM or INT signal."""
    num_idle_processes: int | ServerEnvOption[int] = ServerEnvOption(
        dev_default=0, prod_default=min(math.ceil(get_cpu_monitor().cpu_count()), 4)
    )
    """Number of idle processes to keep warm."""
    shutdown_process_timeout: float = 10.0
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
    ws_url: str | None = None
    """URL to connect to the LiveKit server.

    By default it uses ``LIVEKIT_URL`` from environment"""
    api_key: str | None = None
    """API key to authenticate with LiveKit.

    By default it uses ``LIVEKIT_API_KEY`` from environment"""
    api_secret: str | None = None
    """API secret to authenticate with LiveKit.

    By default it uses ``LIVEKIT_API_SECRET`` from environment"""

    host: str = ""  # default to all interfaces
    port: int | ServerEnvOption[int] = ServerEnvOption(dev_default=0, prod_default=8081)
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
    prometheus_port: NotGivenOr[int] = NOT_GIVEN
    """When enabled, will expose prometheus metrics on :{prometheus_port}/metrics"""
    prometheus_multiproc_dir: str | None = None
    """Directory for prometheus multiprocess mode to enable metrics collection from child job processes.
    When set, the PROMETHEUS_MULTIPROC_DIR environment variable will be configured automatically.
    When None (default), multiprocess mode is disabled and only main process metrics are collected.
    Users can also set PROMETHEUS_MULTIPROC_DIR environment variable directly before starting the worker."""

    def validate_config(self, devmode: bool) -> None:
        load_threshold = ServerEnvOption.getvalue(self.load_threshold, devmode)
        if load_threshold > 1 and not devmode:
            logger.warning(
                f"load_threshold in prod env must be less than 1, current value: {load_threshold}"
            )


WorkerOptions = ServerOptions


@dataclass
class WorkerInfo:
    http_port: int
    cloud_agents: bool


EventTypes = Literal["worker_started", "worker_registered"]


class AgentServer(utils.EventEmitter[EventTypes]):
    _default_num_idle_processes = ServerEnvOption(
        dev_default=0, prod_default=math.ceil(get_cpu_monitor().cpu_count())
    )
    _default_port = ServerEnvOption(dev_default=0, prod_default=8081)

    def __init__(
        self,
        *,
        job_executor_type: JobExecutorType = _default_job_executor_type,
        load_threshold: float | ServerEnvOption[float] = _default_load_threshold,
        job_memory_warn_mb: float = 500,
        job_memory_limit_mb: float = 0,
        drain_timeout: int = 1800,
        num_idle_processes: int | ServerEnvOption[int] = _default_num_idle_processes,
        shutdown_process_timeout: float = 10.0,
        initialize_process_timeout: float = 10.0,
        permissions: WorkerPermissions = _default_permissions,
        max_retry: int = 16,
        ws_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        host: str = "",  # default to all interfaces
        port: int | ServerEnvOption[int] = _default_port,
        http_proxy: NotGivenOr[str | None] = NOT_GIVEN,
        multiprocessing_context: Literal["spawn", "forkserver"] = (
            "spawn" if not sys.platform.startswith("linux") else "forkserver"
        ),
        setup_fnc: Callable[[JobProcess], Any] | None = None,
        load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = None,
        prometheus_port: int | None = None,
    ) -> None:
        super().__init__()
        self._ws_url = ws_url or os.environ.get("LIVEKIT_URL") or ""
        self._api_key = api_key or os.environ.get("LIVEKIT_API_KEY") or ""
        self._api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET") or ""

        self._worker_token = os.environ.get("LIVEKIT_WORKER_TOKEN") or ""  # hosted agents

        self._host = host
        self._port = port
        self._job_executor_type = job_executor_type
        self._load_threshold = load_threshold
        self._job_memory_warn_mb = job_memory_warn_mb
        self._job_memory_limit_mb = job_memory_limit_mb
        self._drain_timeout = drain_timeout
        self._num_idle_processes = num_idle_processes
        self._shutdown_process_timeout = shutdown_process_timeout
        self._initialize_process_timeout = initialize_process_timeout
        self._permissions = permissions
        self._max_retry = max_retry
        self._prometheus_port = prometheus_port
        self._mp_ctx_str = multiprocessing_context
        self._mp_ctx = mp.get_context(multiprocessing_context)

        if not is_given(http_proxy):
            http_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        self._http_proxy = http_proxy
        self._agent_name = ""
        self._server_type = ServerType.ROOM
        self._id = "unregistered"

        # currently only one rtc_session
        self._entrypoint_fnc: Callable[[JobContext], Awaitable[None]] | None = None
        self._request_fnc: Callable[[JobRequest], Awaitable[None]] | None = None
        self._session_end_fnc: Callable[[JobContext], Awaitable[None]] | None = None

        # worker cb
        self._setup_fnc: Callable[[JobProcess], Any] | None = setup_fnc
        self._load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = load_fnc

        self._closed, self._draining, self._connecting = True, False, False
        self._http_server: http_server.HttpServer | None = None

        self._lock = asyncio.Lock()

    if sys.version_info < (3, 10):
        # Python 3.9 cannot pickle asyncio.Lock, customize for pickle support
        def __getstate__(self) -> dict[str, Any]:
            """Custom pickle support - exclude unpickleable asyncio objects."""
            state = self.__dict__.copy()
            # remove unpickleable asyncio.Lock (will be recreated in __setstate__)
            state.pop("_lock", None)
            return state

        def __setstate__(self, state: dict[str, Any]) -> None:
            """Restore state and recreate asyncio.Lock."""
            self.__dict__.update(state)
            # recreate the lock
            self._lock = asyncio.Lock()

    @property
    def setup_fnc(self) -> Callable[[JobProcess], Any] | None:
        return self._setup_fnc

    @setup_fnc.setter
    def setup_fnc(self, value: Callable[[JobProcess], Any] | None) -> None:
        if value is not None and not callable(value):
            raise TypeError("setup_fnc must be a callable or None")
        self._setup_fnc = value

    @property
    def load_fnc(self) -> Callable[[AgentServer], float] | Callable[[], float] | None:
        return self._load_fnc

    @load_fnc.setter
    def load_fnc(self, value: Callable[..., float] | None) -> None:
        if value is not None and not callable(value):
            raise TypeError("load_fnc must be a callable or None")
        self._load_fnc = value

    @classmethod
    def from_server_options(cls, options: ServerOptions) -> AgentServer:
        server = cls(
            job_executor_type=options.job_executor_type,
            load_threshold=options.load_threshold,
            job_memory_limit_mb=options.job_memory_limit_mb,
            job_memory_warn_mb=options.job_memory_warn_mb,
            drain_timeout=options.drain_timeout,
            num_idle_processes=options.num_idle_processes,
            shutdown_process_timeout=options.shutdown_process_timeout,
            initialize_process_timeout=options.initialize_process_timeout,
            permissions=options.permissions,
            max_retry=options.max_retry,
            ws_url=options.ws_url,
            api_key=options.api_key,
            api_secret=options.api_secret,
            host=options.host,
            port=options.port,
            http_proxy=options.http_proxy,
            multiprocessing_context=options.multiprocessing_context,
            prometheus_port=options.prometheus_port if is_given(options.prometheus_port) else None,
            setup_fnc=options.prewarm_fnc,
            load_fnc=options.load_fnc,
        )
        server.rtc_session(
            options.entrypoint_fnc,
            agent_name=options.agent_name,
            type=options.worker_type,
            on_request=options.request_fnc,
        )
        return server

    @overload
    def rtc_session(
        self,
        func: Callable[[JobContext], Awaitable[None]],
        *,
        agent_name: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
    ) -> Callable[[JobContext], Awaitable[None]]: ...

    @overload
    def rtc_session(
        self,
        *,
        agent_name: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
    ) -> Callable[
        [Callable[[JobContext], Awaitable[None]]], Callable[[JobContext], Awaitable[None]]
    ]: ...

    def rtc_session(
        self,
        func: Callable[[JobContext], Awaitable[None]] | None = None,
        *,
        agent_name: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
    ) -> (
        Callable[[JobContext], Awaitable[None]]
        | Callable[
            [Callable[[JobContext], Awaitable[None]]], Callable[[JobContext], Awaitable[None]]
        ]
    ):
        """
        Decorator or direct registrar for the RTC session entrypoint.

        Usage:
            @server.rtc_session(agent_name="survey_agent")
            async def my_agent(job_ctx: JobContext): ...

            server.rtc_session(my_agent, agent_name="survey_agent")
        """

        def decorator(
            f: Callable[[JobContext], Awaitable[None]],
        ) -> Callable[[JobContext], Awaitable[None]]:
            if self._entrypoint_fnc is not None:
                raise RuntimeError(
                    "The AgentServer currently only supports registering only one rtc_session"
                )
            self._entrypoint_fnc = f
            self._request_fnc = on_request
            self._session_end_fnc = on_session_end
            self._agent_name = agent_name
            self._server_type = type
            return f

        if func is not None:
            return decorator(func)

        return decorator

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(
            http_port=self._http_server.port if self._http_server else 0,
            cloud_agents=bool(self._worker_token),
        )

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> None:
        """This method starts the worker's internal event loop, initializes any required
        executors, HTTP servers, and process pools, and optionally registers the worker
        with the LiveKit server.

        Args:
            devmode (bool, optional):
                If True, the worker runs in development mode.
                This affects certain environment-dependent defaults, such as the
                number of idle processes, logging verbosity, and load thresholds,
                making it easier to test and debug without production constraints.

            unregistered (bool, optional):
                If True, the worker will start without registering itself with the
                LiveKit server.
                This allows the worker to operate in a partially connected state—
                capable of using other providers or local processing—but invisible
                to the central LiveKit job dispatcher.
                Useful for local testing, isolated jobs, or running without being
                assigned new jobs.
        """
        async with self._lock:
            if not self._closed:
                raise Exception("worker is already running")

            if self._entrypoint_fnc is None:
                raise RuntimeError(
                    "No RTC session entrypoint has been registered.\n"
                    "Define one using the @server.rtc_session() decorator, for example:\n"
                    '    @server.rtc_session(agent_name="my_agent")\n'
                    "    async def my_agent(ctx: JobContext):\n"
                    "        ...\n"
                )

            if self._request_fnc is None:
                self._request_fnc = _default_request_fnc

            if self._setup_fnc is None:
                self._setup_fnc = _default_setup_fnc

            if self._load_fnc is None:
                self._load_fnc = _DefaultLoadCalc.get_load

            if self.worker_info.cloud_agents:
                if self._load_fnc != _DefaultLoadCalc.get_load:
                    logger.warning(
                        "custom load_fnc is not supported when hosting on Cloud, reverting to default"
                    )
                    self._load_fnc = _DefaultLoadCalc.get_load
                if self._load_threshold != _default_load_threshold:
                    logger.warning(
                        "custom load_threshold is not supported when hosting on Cloud, reverting to default"
                    )
                    self._load_threshold = _default_load_threshold

            self._loop = asyncio.get_event_loop()
            self._devmode = devmode
            self._tasks = set[asyncio.Task[Any]]()
            self._pending_assignments: dict[str, asyncio.Future[agent.JobAssignment]] = {}
            self._close_future: asyncio.Future[None] | None = None
            self._msg_chan = utils.aio.Chan[agent.WorkerMessage](128, loop=self._loop)

            self._inference_executor: ipc.inference_proc_executor.InferenceProcExecutor | None = (
                None
            )
            if len(_InferenceRunner.registered_runners) > 0:
                self._inference_executor = ipc.inference_proc_executor.InferenceProcExecutor(
                    runners=_InferenceRunner.registered_runners,
                    initialize_timeout=5 * 60,
                    close_timeout=5,
                    memory_warn_mb=2000,
                    memory_limit_mb=0,  # no limit
                    ping_interval=5,
                    ping_timeout=60,
                    high_ping_threshold=2.5,
                    mp_ctx=self._mp_ctx,
                    loop=self._loop,
                    http_proxy=self._http_proxy or None,
                )

            self._proc_pool = ipc.proc_pool.ProcPool(
                initialize_process_fnc=self._setup_fnc,
                job_entrypoint_fnc=self._entrypoint_fnc,
                session_end_fnc=self._session_end_fnc,
                num_idle_processes=ServerEnvOption.getvalue(self._num_idle_processes, devmode),
                loop=self._loop,
                job_executor_type=self._job_executor_type,
                inference_executor=self._inference_executor,
                mp_ctx=self._mp_ctx,
                initialize_timeout=self._initialize_process_timeout,
                close_timeout=self._shutdown_process_timeout,
                memory_warn_mb=self._job_memory_warn_mb,
                memory_limit_mb=self._job_memory_limit_mb,
                http_proxy=self._http_proxy or None,
            )

            self._previous_status = agent.WorkerStatus.WS_AVAILABLE

            self._api: api.LiveKitAPI | None = None
            self._http_session: aiohttp.ClientSession | None = None
            self._http_server = http_server.HttpServer(
                self._host, ServerEnvOption.getvalue(self._port, devmode), loop=self._loop
            )
            self._worker_load: float = 0.0

            async def health_check(_: Any) -> web.Response:
                if self._inference_executor and not self._inference_executor.is_alive():
                    return web.Response(status=503, text="inference process not running")

                return web.Response(text="OK")

            async def worker(_: Any) -> web.Response:
                body = json.dumps(
                    {
                        "agent_name": self._agent_name,
                        "worker_type": agent.JobType.Name(self._server_type.value),
                        "worker_load": self._worker_load,
                        "active_jobs": len(self.active_jobs),
                        "sdk_version": __version__,
                        "project_type": "python",
                    }
                )
                return web.Response(body=body, content_type="application/json")

            self._http_server.app.add_routes([web.get("/", health_check)])
            self._http_server.app.add_routes([web.get("/worker", worker)])

            self._prometheus_server: telemetry.http_server.HttpServer | None = None
            if self._prometheus_port is not None:
                self._prometheus_server = telemetry.http_server.HttpServer(
                    self._host, self._prometheus_port, loop=self._loop
                )

            self._conn_task: asyncio.Task[None] | None = None
            self._load_task: asyncio.Task[None] | None = None

            if not self._ws_url:
                raise ValueError("ws_url is required, or add LIVEKIT_URL in your environment")

            if not self._api_key:
                raise ValueError("api_key is required, or add LIVEKIT_API_KEY in your environment")

            if not self._api_secret:
                raise ValueError(
                    "api_secret is required, or add LIVEKIT_API_SECRET in your environment"
                )

            os.environ["LIVEKIT_URL"] = self._ws_url
            os.environ["LIVEKIT_API_KEY"] = self._api_key
            os.environ["LIVEKIT_API_SECRET"] = self._api_secret

            logger.info(
                "starting worker",
                extra={"version": __version__, "rtc-version": rtc.__version__},
            )

            if self._mp_ctx_str == "forkserver":
                plugin_packages = [p.package for p in Plugin.registered_plugins] + ["av"]
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

            if self._prometheus_server:
                await self._prometheus_server.start()

            self._proc_pool.on("process_started", _update_job_status)
            self._proc_pool.on("process_closed", _update_job_status)
            self._proc_pool.on("process_job_launched", _update_job_status)
            await self._proc_pool.start()

            self._http_session = aiohttp.ClientSession(proxy=self._http_proxy or None)
            self._api = api.LiveKitAPI(
                self._ws_url, self._api_key, self._api_secret, session=self._http_session
            )
            self._close_future = asyncio.Future(loop=self._loop)

            @utils.log_exceptions(logger=logger)
            async def _load_task() -> None:
                """periodically check load"""

                interval = utils.aio.interval(UPDATE_LOAD_INTERVAL)
                while True:
                    await interval.tick()

                    def load_fnc() -> float:
                        assert self._load_fnc is not None
                        signature = inspect.signature(self._load_fnc)
                        parameters = list(signature.parameters.values())
                        if len(parameters) == 0:
                            return self._load_fnc()  # type: ignore

                        return self._load_fnc(self)  # type: ignore

                    self._worker_load = await asyncio.get_event_loop().run_in_executor(
                        None, load_fnc
                    )

                    telemetry.metrics._update_worker_load(self._worker_load)

                    load_threshold = ServerEnvOption.getvalue(self._load_threshold, devmode)
                    default_num_idle_processes = ServerEnvOption.getvalue(
                        self._num_idle_processes, devmode
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

            tasks = []
            self._load_task = asyncio.create_task(_load_task(), name="load_task")
            tasks.append(self._load_task)

            if not unregistered:
                self._conn_task = asyncio.create_task(
                    self._connection_task(), name="worker_conn_task"
                )
                tasks.append(self._conn_task)

            self.emit("worker_started")

        await self._close_future

    def update_options(
        self,
        *,
        ws_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        max_retry: NotGivenOr[int] = NOT_GIVEN,
        job_executor_type: NotGivenOr[JobExecutorType] = NOT_GIVEN,
        load_threshold: NotGivenOr[float] = NOT_GIVEN,
        job_memory_warn_mb: NotGivenOr[float] = NOT_GIVEN,
        job_memory_limit_mb: NotGivenOr[float] = NOT_GIVEN,
        drain_timeout: NotGivenOr[int] = NOT_GIVEN,
        num_idle_processes: NotGivenOr[int] = NOT_GIVEN,
        shutdown_process_timeout: float = 10.0,
        initialize_process_timeout: float = 10.0,
    ) -> None:
        if not self._closed:
            raise RuntimeError("cannot update options after starting the server")

        if is_given(ws_url):
            self._ws_url = ws_url

        if is_given(api_key):
            self._api_key = api_key

        if is_given(api_secret):
            self._api_secret = api_secret

        if is_given(max_retry):
            self._max_retry = max_retry

        if is_given(job_executor_type):
            self._job_executor_type = job_executor_type

        if is_given(load_threshold):
            self._load_threshold = load_threshold

        if is_given(job_memory_warn_mb):
            self._job_memory_warn_mb = job_memory_warn_mb

        if is_given(job_memory_limit_mb):
            self._job_memory_limit_mb = job_memory_limit_mb

        if is_given(drain_timeout):
            self._drain_timeout = drain_timeout

        if is_given(num_idle_processes):
            self._num_idle_processes = num_idle_processes

        if is_given(shutdown_process_timeout):
            self._shutdown_process_timeout = shutdown_process_timeout

    @property
    def id(self) -> str:
        return self._id

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return [proc.running_job for proc in self._proc_pool.processes if proc.running_job]

    async def drain(self, timeout: NotGivenOr[int | None] = NOT_GIVEN) -> None:
        """When timeout isn't None, it will raise asyncio.TimeoutError if the processes didn't finish in time."""  # noqa: E501

        timeout = timeout if is_given(timeout) else self._drain_timeout

        async with self._lock:
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
                await asyncio.wait_for(
                    _join_jobs(), timeout
                )  # raises asyncio.TimeoutError on timeout
            else:
                await _join_jobs()

    @utils.log_exceptions(logger=logger)
    async def simulate_job(
        self,
        room: str,
        *,
        fake_job: bool = False,
        agent_identity: str | None = None,
        room_info: models.Room | None = None,
        token: str | None = None,
    ) -> None:
        async with self._lock:
            if token is not None:
                # read identity from token if provided
                agent_identity = api.TokenVerifier().verify(token, verify_signature=False).identity

            if agent_identity is None:
                if not fake_job:
                    raise ValueError("agent_identity is None but fake_job is False")

                agent_identity = utils.shortuuid("fake-agent-")

            if room_info is None:
                if not fake_job:
                    raise ValueError("room_info is None but fake_job is False")

                room_info = models.Room(sid=utils.shortuuid("FAKE_RM_"), name=room)

            # room_info = await self._api.room.create_room(api.CreateRoomRequest(name=room))

            job = agent.Job(
                id=utils.shortuuid("simulated-job-")
                if not fake_job
                else utils.shortuuid("fake-job-"),
                room=room_info,
                type=agent.JobType.JT_ROOM,
                participant=None,
            )

            token = token or (
                api.AccessToken(self._api_key, self._api_secret)
                .with_identity(agent_identity)
                .with_kind("agent")
                .with_grants(api.VideoGrants(room_join=True, room=room, agent=True))
                .to_jwt()
            )
            running_info = RunningJobInfo(
                worker_id=self._id,
                accept_arguments=JobAcceptArguments(identity=agent_identity, name="", metadata=""),
                job=job,
                url=self._ws_url,
                token=token,
                fake_job=fake_job,
            )

            await self._proc_pool.launch_job(running_info)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("cannot simulate job, the worker is closed")

            if self._closed:
                if self._close_future is not None:
                    await self._close_future
                return

            logger.info("shutting down worker", extra={"id": self.id})

            assert self._close_future is not None
            assert self._http_session is not None
            assert self._api is not None
            assert self._http_server is not None

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

            if self._prometheus_server:
                await self._prometheus_server.aclose()

            await self._api.aclose()  # type: ignore

            await asyncio.gather(*self._tasks, return_exceptions=True)

            # await asyncio.sleep(0.25)  # see https://github.com/aio-libs/aiohttp/issues/1925
            self._msg_chan.close()

            if not self._close_future.done():
                self._close_future.set_result(None)

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
                    api.AccessToken(self._api_key, self._api_secret)
                    .with_grants(api.VideoGrants(agent=True))
                    .to_jwt()
                )

                headers = {"Authorization": f"Bearer {join_jwt}"}

                parse = urlparse(self._ws_url)
                scheme = parse.scheme
                if scheme.startswith("http"):
                    scheme = scheme.replace("http", "ws")

                base = f"{scheme}://{parse.netloc}{parse.path}".rstrip("/") + "/"
                agent_url = urljoin(base, "agent")

                params = {}
                if self._worker_token:
                    params["worker_token"] = self._worker_token

                ws = await self._http_session.ws_connect(
                    agent_url,
                    headers=headers,
                    params=params,
                    autoping=True,
                    proxy=self._http_proxy or None,
                    heartbeat=HEARTBEAT_INTERVAL,
                )

                retry_count = 0

                # register the worker
                req = agent.WorkerMessage()
                req.register.type = self._server_type.value
                req.register.allowed_permissions.CopyFrom(
                    models.ParticipantPermission(
                        can_publish=self._permissions.can_publish,
                        can_subscribe=self._permissions.can_subscribe,
                        can_publish_data=self._permissions.can_publish_data,
                        can_update_metadata=self._permissions.can_update_metadata,
                        can_publish_sources=self._permissions.can_publish_sources,
                        hidden=self._permissions.hidden,
                        agent=True,
                    )
                )
                req.register.agent_name = self._agent_name
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

                if retry_count >= self._max_retry:
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
        if not self._api_secret:
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
            decoded = jwt.decode(original_token, self._api_secret, algorithms=["HS256"])
            decoded["exp"] = int(datetime.datetime.now(datetime.timezone.utc).timestamp()) + 3600
            running_info = RunningJobInfo(
                accept_arguments=aj.accept_arguments,
                job=aj.job,
                url=self._ws_url,
                token=jwt.encode(decoded, self._api_secret, algorithm="HS256"),
                worker_id=aj.worker_id,
                fake_job=aj.fake_job,
            )
            await self._proc_pool.launch_job(running_info)

    def _handle_register(self, reg: agent.RegisterWorkerResponse) -> None:
        self._id = reg.worker_id
        logger.info(
            "registered worker",
            extra={
                "agent_name": self._agent_name,
                "id": reg.worker_id,
                "url": self._ws_url,
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
                    extra={"job_request": job_req, "agent_name": self._agent_name},
                )
                raise AssignmentTimeoutError() from None

            job_assign = wait_assignment.result()
            running_info = RunningJobInfo(
                accept_arguments=args,
                job=msg.job,
                url=job_assign.url or self._ws_url,
                token=job_assign.token,
                worker_id=self._id,
                fake_job=False,
            )

            await self._proc_pool.launch_job(running_info)

        job_req = JobRequest(job=msg.job, on_reject=_on_reject, on_accept=_on_accept)

        logger.info(
            "received job request",
            extra={
                "job_id": msg.job.id,
                "dispatch_id": msg.job.dispatch_id,
                "room": msg.job.room.name,
                "room_id": msg.job.room.sid,
                "agent_name": self._agent_name,
                "resuming": msg.resuming,
                "enable_recording": msg.job.enable_recording,
            },
        )

        @utils.log_exceptions(logger=logger)
        async def _job_request_task() -> None:
            assert self._request_fnc is not None
            try:
                await self._request_fnc(job_req)
            except Exception:
                logger.exception(
                    "job_request_fnc failed",
                    extra={"job_request": job_req, "agent_name": self._agent_name},
                )

            if not answered:
                logger.warning(
                    "no answer was given inside the job_request_fnc, automatically rejecting the job",  # noqa: E501
                    extra={"job_request": job_req, "agent_name": self._agent_name},
                )
                await _on_reject()

        user_task = self._loop.create_task(_job_request_task(), name="job_request")
        self._tasks.add(user_task)
        user_task.add_done_callback(self._tasks.discard)

    def _handle_assignment(self, assignment: agent.JobAssignment) -> None:
        logger.debug(
            "received assignment",
            extra={
                "agent_name": self._agent_name,
                "room_id": assignment.job.room.sid,
                "room": assignment.job.room.name,
                "job_id": assignment.job.id,
                "dispatch_id": assignment.job.dispatch_id,
                "enable_recording": assignment.job.enable_recording,
            },
        )
        if assignment.job.id in self._pending_assignments:
            with contextlib.suppress(asyncio.InvalidStateError):
                fut = self._pending_assignments.pop(assignment.job.id)
                fut.set_result(assignment)
        else:
            logger.warning(
                "received assignment for an unknown job",
                extra={"job": MessageToDict(assignment.job), "agent_name": self._agent_name},
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

        load_threshold = ServerEnvOption.getvalue(self._load_threshold, self._devmode)
        is_full = self._worker_load >= load_threshold
        currently_available = not is_full and not self._draining

        status = (
            agent.WorkerStatus.WS_AVAILABLE if currently_available else agent.WorkerStatus.WS_FULL
        )

        update = agent.UpdateWorkerStatus(load=self._worker_load, status=status, job_count=job_cnt)

        # only log if status has changed
        if self._previous_status != status and not self._draining:
            self._previous_status = status
            extra = {"load": self._worker_load, "threshold": load_threshold}
            if is_full:
                logger.info("worker is at full capacity, marking as unavailable", extra=extra)
            else:
                logger.info("worker is below capacity, marking as available", extra=extra)

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
