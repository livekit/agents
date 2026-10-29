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
import dataclasses
import datetime
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import aiohttp
import jwt
from aiohttp import web
from google.protobuf.json_format import MessageToJson
from starlette.types import ASGIApp

from livekit import api, rtc
from livekit.protocol import agent, agent_worker, models

from . import _core, ipc, telemetry, utils
from ._generated.storage_turn import Policy
from .http import AgentHttp
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
from .simulation import SimulationContext
from .storage.agentdb import AgentDBStorage
from .storage.context import Handler, S
from .storage.protocol import StorageError
from .storage.resolve import AgentRegistry, Thread, resolve_thread
from .types import NOT_GIVEN, NotGivenOr
from .utils import http_context, http_server, is_given
from .version import __version__

if TYPE_CHECKING:
    from .a2a.endpoint import A2AEndpoint, AuthCallable

# The control plane (registration, availability, assignment, status, drain) and the endpoint data
# plane are the Rust worker's, `_core.Worker`; this module owns the process pool, the health
# listener and `AgentServer.http`. Timing constants live in the core.
UPDATE_LOAD_INTERVAL = 0.5
WORKER_PROTOCOL_VERSION = 1
DRAIN_TIMEOUT = 3600  # 1hr


@dataclass(frozen=True)
class _CallThreadEntrypoint:
    """The ``rtc_session`` entrypoint of a server with storage: the call's thread is resolved
    first, the ``thread_id`` of the dispatch metadata, else a new thread, as ``ctx.thread``.
    A storage failure is logged and the call runs with no thread."""

    entrypoint: Callable[[JobContext], Awaitable[None]]
    storage: AgentDBStorage
    call_holds_lane: bool = True
    """The mode the lane's text door gives the call (``Thread.call_holds_lane``)."""

    async def __call__(self, ctx: JobContext) -> None:
        if not self.storage.holder:
            self.storage.holder = ctx.worker_id
        try:
            metadata = json.loads(ctx.job.metadata or "{}")
        except ValueError:
            metadata = None
        thread_id = metadata.get("thread_id") if isinstance(metadata, dict) else None
        try:
            if not isinstance(thread_id, str) or not thread_id:
                thread_id = (await resolve_thread(self.storage.threads)).id
            thread = Thread(self.storage, thread_id, agent_name=ctx.job.agent_name)
            thread._call_holds_lane = self.call_holds_lane
            ctx._thread = thread
        except StorageError as e:
            logger.warning(
                "the call's thread could not be resolved; the call is kept in memory",
                extra={"threadId": thread_id, "error": str(e)},
            )
        await self.entrypoint(ctx)


async def _text_only_entrypoint(ctx: JobContext) -> None:
    """The entrypoint of a server with text doors and no voice door: a job is not expected."""
    logger.warning(
        "a job reached a server with no rtc_session; nothing joins the room",
        extra={"room": ctx.room.name},
    )


def _default_setup_fnc(proc: JobProcess) -> Any:
    return


async def _default_request_fnc(ctx: JobRequest) -> None:
    await ctx.accept()


class ServerType(Enum):
    ROOM = agent.JobType.JT_ROOM
    PUBLISHER = agent.JobType.JT_PUBLISHER


WorkerType = ServerType


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
_default_log_level = ServerEnvOption(dev_default="DEBUG", prod_default="INFO")
_default_permissions = WorkerPermissions()

VALID_LOG_LEVELS = frozenset({"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"})


def _validate_and_normalize_log_level(
    log_level: str | ServerEnvOption[str],
) -> str | ServerEnvOption[str]:
    if isinstance(log_level, ServerEnvOption):
        levels_to_check = [log_level.dev_default, log_level.prod_default]
    else:
        levels_to_check = [log_level]

    for level in levels_to_check:
        if level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level {level!r}. Valid levels: {', '.join(sorted(VALID_LOG_LEVELS))}"
            )

    if isinstance(log_level, ServerEnvOption):
        return ServerEnvOption(
            dev_default=log_level.dev_default.upper(),
            prod_default=log_level.prod_default.upper(),
        )
    return log_level.upper()


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
    load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = None
    """Called to determine the current load of the worker. Should return a value between 0 and 1.
    Left ``None``, the worker samples the CPU itself (in the Rust core)."""
    job_executor_type: JobExecutorType = _default_job_executor_type
    """Which executor to use to run jobs. (currently thread or process are supported)"""
    load_threshold: float | ServerEnvOption[float] = _default_load_threshold
    """When the load exceeds this threshold, the worker will be marked as unavailable.

    Defaults to 0.7 on "production" mode, and is disabled in "development" mode.
    """

    job_memory_warn_mb: float = 1000
    """Memory warning threshold in MB. If the job process exceeds this limit, a warning will be logged."""  # noqa: E501
    job_memory_limit_mb: float = 0
    """Maximum memory usage for a job in MB, the job process will be killed if it exceeds this limit.
    Defaults to 0 (disabled).
    """  # noqa: E501

    drain_timeout: int = DRAIN_TIMEOUT
    """Number of seconds to wait for current jobs to finish upon receiving TERM or INT signal."""
    num_idle_processes: int | ServerEnvOption[int] = ServerEnvOption(
        dev_default=0, prod_default=min(math.ceil(_core.cpu_count()), 4)
    )
    """Number of idle processes to keep warm."""
    shutdown_process_timeout: float = 10.0
    """Maximum amount of time to wait for a job to shut down gracefully"""
    session_end_timeout: float = 300.0
    """Maximum amount of time to wait for on_session_end to complete (default: 5 minutes)."""
    initialize_process_timeout: float = 10.0
    """Maximum amount of time to wait for a process to initialize/prewarm"""
    permissions: WorkerPermissions = field(default_factory=WorkerPermissions)
    """Permissions that the agent should join the room with."""
    agent_name: str = ""
    """Set agent_name to enable explicit dispatch. When explicit dispatch is enabled, jobs will not be dispatched to rooms automatically. Instead, you can either specify the agent(s) to be dispatched in the end-user's token, or use the AgentDispatch.createDispatch API.

    By default it uses ``LIVEKIT_AGENT_NAME`` from environment"""  # noqa: E501
    worker_type: WorkerType = WorkerType.ROOM
    """Whether to spin up an agent for each room or publisher."""
    max_retry: int = 16
    """Maximum number of times to retry connecting to LiveKit."""
    ws_url: str | None = None
    """URL to connect to the LiveKit server.

    By default it uses ``LIVEKIT_URL`` from environment"""
    api_key: str | None = field(repr=False, default=None)
    """API key to authenticate with LiveKit.

    By default it uses ``LIVEKIT_API_KEY`` from environment"""
    api_secret: str | None = field(repr=False, default=None)
    """API secret to authenticate with LiveKit.

    By default it uses ``LIVEKIT_API_SECRET`` from environment"""

    log_level: str | ServerEnvOption[str] = _default_log_level
    """Log level for the worker.

    Defaults to ``DEBUG`` in development mode and ``INFO`` in production mode.
    Can also be set via the ``LIVEKIT_LOG_LEVEL`` environment variable or
    the ``--log-level`` CLI argument (CLI takes highest precedence)."""

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

    def __post_init__(self) -> None:
        self.log_level = _validate_and_normalize_log_level(self.log_level)

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


@dataclass(frozen=True)
class RtcDoor:
    """The voice door as an object: what ``rtc_session`` returns and ``AgentServer.add`` serves."""

    entrypoint: Callable[[JobContext], Awaitable[None]]
    endpoint: str = ""
    type: ServerType = ServerType.ROOM
    on_request: Callable[[JobRequest], Any] | None = None
    on_session_end: Callable[[JobContext], Any] | None = None
    on_simulation_end: Callable[[SimulationContext], Any] | None = None


def rtc_session(
    *,
    endpoint: str = "",
    type: ServerType = ServerType.ROOM,
    on_request: Callable[[JobRequest], Any] | None = None,
    on_session_end: Callable[[JobContext], Any] | None = None,
    on_simulation_end: Callable[[SimulationContext], Any] | None = None,
) -> Callable[[Callable[[JobContext], Awaitable[None]]], RtcDoor]:
    """The voice door as a decorator with no server: a module defines it, ``AgentServer.add()``
    serves it (``server.rtc_session()`` does both)."""

    def decorator(f: Callable[[JobContext], Awaitable[None]]) -> RtcDoor:
        return RtcDoor(
            f,
            endpoint=endpoint,
            type=type,
            on_request=on_request,
            on_session_end=on_session_end,
            on_simulation_end=on_simulation_end,
        )

    return decorator


_RtcEntrypointT = TypeVar("_RtcEntrypointT", bound=Callable[[JobContext], Awaitable[None]])


class AgentHandler:
    """An agent's doors on one lane: what a module defines and ``AgentServer.add`` serves.

    ::

        back_office = AgentHandler(lane="backend")

        @back_office.a2a_session(endpoint="backend", description="The back office.", public=False)
        async def text(ctx: A2ASessionContext) -> str: ...

    ``lane`` names the lane the doors write (docs/lanes.md); the worker's own when left out.
    The voice door writes the worker's lane, so a handler with a lane of its own has text
    doors only.
    """

    def __init__(self, *, lane: str | None = None) -> None:
        self.lane = lane
        self.doors: list[RtcDoor | A2AEndpoint[Any]] = []

    @overload
    def rtc_session(
        self,
        func: _RtcEntrypointT,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> _RtcEntrypointT: ...

    @overload
    def rtc_session(
        self,
        func: None = None,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> Callable[[_RtcEntrypointT], _RtcEntrypointT]: ...

    def rtc_session(
        self,
        func: Callable[[JobContext], Awaitable[None]] | None = None,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> Any:
        """The voice door, as ``AgentServer.rtc_session``."""
        door = rtc_session(
            endpoint=endpoint,
            type=type,
            on_request=on_request,
            on_session_end=on_session_end,
            on_simulation_end=on_simulation_end,
        )

        def decorator(f: Callable[[JobContext], Awaitable[None]]) -> Any:
            if self.lane:
                raise ValueError(
                    "the voice door writes the worker's lane: AgentHandler() without lane="
                )
            self.doors.append(door(f))
            return f

        return decorator(func) if func is not None else decorator

    def a2a_session(
        self,
        *,
        endpoint: str,
        description: str,
        state: type[S] | None = None,
        public: bool = True,
        auth: AuthCallable | None = None,
        inbox: Policy | Mapping[str, Policy] | None = None,
        debounce: datetime.timedelta | Mapping[str, datetime.timedelta] | None = None,
        storage: AgentDBStorage | None = None,
        version: str | None = None,
    ) -> Callable[[Handler[S]], A2AEndpoint[S]]:
        """The text door, as ``AgentServer.a2a_session``, on this handler's lane."""
        from .a2a.endpoint import a2a_session

        door = a2a_session(
            endpoint=endpoint,
            description=description,
            state=state,
            public=public,
            auth=auth,
            inbox=inbox,
            debounce=debounce,
            storage=storage,
            version=version,
            lane=self.lane,
        )

        def decorator(handler: Handler[S]) -> A2AEndpoint[S]:
            ep = door(handler)
            self.doors.append(ep)
            return ep

        return decorator


class AgentServer(utils.EventEmitter[EventTypes]):
    _default_num_idle_processes = ServerEnvOption(
        dev_default=0, prod_default=math.ceil(_core.cpu_count())
    )
    _default_port = ServerEnvOption(dev_default=0, prod_default=8081)

    def __init__(
        self,
        *,
        job_executor_type: JobExecutorType = _default_job_executor_type,
        load_threshold: float | ServerEnvOption[float] = _default_load_threshold,
        job_memory_warn_mb: float = 1000,
        job_memory_limit_mb: float = 0,
        drain_timeout: int = DRAIN_TIMEOUT,
        num_idle_processes: int | ServerEnvOption[int] = _default_num_idle_processes,
        shutdown_process_timeout: float = 10.0,
        session_end_timeout: float = 300.0,
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
        prometheus_multiproc_dir: str | None = None,
        log_level: str | ServerEnvOption[str] = _default_log_level,
        storage: AgentDBStorage | None = None,
        agent_name: str = "",
    ) -> None:
        super().__init__()
        # the session storage every door of this server opens its lane with (plan 5.7), and
        # the agents a stored session may name (plan 5.3 rule 9)
        self._storage = storage
        self._agents = AgentRegistry()
        self._ws_url = ws_url or os.environ.get("LIVEKIT_URL") or ""
        self._api_key = api_key or os.environ.get("LIVEKIT_API_KEY") or ""
        self._api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET") or ""

        self._worker_token = os.environ.get("LIVEKIT_WORKER_TOKEN") or ""  # hosted agents
        self._deployment = os.environ.get("LIVEKIT_AGENT_DEPLOYMENT") or ""  # hosted agents

        self._host = host
        self._port = port
        self._job_executor_type = job_executor_type
        self._load_threshold = load_threshold
        self._job_memory_warn_mb = job_memory_warn_mb
        self._job_memory_limit_mb = job_memory_limit_mb
        self._drain_timeout = drain_timeout
        self._num_idle_processes = num_idle_processes
        self._shutdown_process_timeout = shutdown_process_timeout
        self._session_end_timeout = session_end_timeout
        self._initialize_process_timeout = initialize_process_timeout
        self._permissions = permissions
        self._max_retry = max_retry
        self._prometheus_port = prometheus_port
        self._prometheus_multiproc_dir = prometheus_multiproc_dir
        self._mp_ctx_str = multiprocessing_context
        self._mp_ctx = mp.get_context(multiprocessing_context)

        if not is_given(http_proxy):
            http_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        self._http_proxy = http_proxy
        self._log_level = _validate_and_normalize_log_level(log_level)
        # Set by the CLI (--simulation) when the worker runs under an agent
        # simulation: load shedding is disabled so runs can saturate the agent.
        self._simulation = False
        # the agent this server is: jobs are dispatched to it, and its doors are reached at
        # /agents/{agent_name}/{deployment}/{endpoint}. A launcher's LIVEKIT_AGENT_NAME_OVERRIDE
        # wins, then the argument, then LIVEKIT_AGENT_NAME.
        self._agent_name = (
            os.environ.get("LIVEKIT_AGENT_NAME_OVERRIDE")
            or agent_name
            or os.environ.get("LIVEKIT_AGENT_NAME", "")
        )
        self._rtc_endpoint = ""
        self._server_type = ServerType.ROOM
        self._id = "unregistered"

        # currently only one rtc_session
        self._entrypoint_fnc: Callable[[JobContext], Awaitable[None]] | None = None
        self._request_fnc: Callable[[JobRequest], Awaitable[None]] | None = None
        self._session_end_fnc: Callable[[JobContext], Awaitable[None]] | None = None
        self._simulation_end_fnc: Callable[[SimulationContext], Any] | None = None

        # worker cb
        self._setup_fnc: Callable[[JobProcess], Any] | None = setup_fnc
        self._load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = load_fnc

        self._closed, self._draining, self._connection_failed = True, False, False
        self._http_server: http_server.HttpServer | None = None
        # built here and not in run(): the @server.http decorators run at import time
        self._http = AgentHttp()
        self._a2a_endpoints: dict[str, A2AEndpoint[Any]] = {}
        self._conn: _core.Worker | None = None  # once run() has started the Rust worker

        # the call's mode follows the person's text door on its lane: under live, or with no
        # text door, the call holds the lane; otherwise it takes it per voice turn
        self._call_holds_lane = True

        self._lock = asyncio.Lock()

    @property
    def storage(self) -> AgentDBStorage | None:
        """The session storage every door of this server opens its lane with (plan 5.7)."""
        return self._storage

    @property
    def agents(self) -> AgentRegistry:
        """The registered agents and task classes: what a stored ``Ref`` rebuilds (plan 5.3)."""
        return self._agents

    def register(self, *agents: Any) -> None:
        """Register ``Agent`` instances or ``AgentTask`` classes a stored session may name.

        Every agent reachable from the server's endpoints is registered when they are; this
        adds the rest, such as the targets of handoffs made inside tools.
        """
        self._agents.register_all(agents)

    @property
    def http(self) -> AgentHttp:
        """HTTP routes served alongside the agent, as a Starlette application::

            @server.http.get("/token")
            async def token(request: Request) -> Response:
                return JSONResponse({"ok": True})

        Routes are served through the LiveKit server at
        ``/agents/{agent_name}/{deployment}/{path}``, over the agent endpoint data plane; no
        local listener exposes them. Serving them requires an ``agent_name``. The handler reads
        the request's preamble as ``request.state.lk_request``.
        """
        return self._http

    def mount(self, path: str, app: ASGIApp, *, name: str | None = None) -> None:
        """Serve an ASGI application (FastAPI, another Starlette app, ...) under ``path``."""
        self._http.mount(path, app, name=name)

    def a2a_session(
        self,
        *,
        endpoint: str,
        description: str,
        state: type[S] | None = None,
        public: bool = True,
        auth: AuthCallable | None = None,
        inbox: Policy | Mapping[str, Policy] | None = None,
        debounce: datetime.timedelta | Mapping[str, datetime.timedelta] | None = None,
        storage: AgentDBStorage | None = None,
        version: str | None = None,
        lane: str | None = None,
    ) -> Callable[[Handler[S]], A2AEndpoint[S]]:
        """The text door (plan 7.3): one A2A endpoint served as an ASGI app under ``/<endpoint>``.

        ::

            @server.a2a_session(endpoint="scheduling", description="Book or move an appointment.", state=Patient)
            async def scheduling(ctx: A2ASessionContext[Patient]) -> str:
                result = await scheduler.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
                return str(result.output)

        The endpoint publishes its card at ``/<endpoint>/.well-known/agent-card.json`` and
        serves JSON-RPC over HTTP+SSE and over WebSocket at ``/<endpoint>/v1``. The route is
        a session route: the node resolves the thread and hands it over in the preamble.
        ``livekit.agents.a2a_session`` is the same decorator with no server, for a door a
        module defines and ``add()`` serves.

        Args:
            endpoint: The last path segment of the endpoint's URL.
            description: What the card says the agent does.
            state: The door's state type, loaded per thread.
            public: Whether a caller needs no token.
            auth: A callable deciding who may call.
            inbox: The turn policy for a message that arrives while one runs.
            debounce: How long to wait for more of a burst.
            storage: This endpoint's storage when it is not the server's.
            version: The card's version; the deployment's by default.
            lane: The lane the door writes, when not the worker's own.
        """
        from .a2a.endpoint import a2a_session

        door = a2a_session(
            endpoint=endpoint,
            description=description,
            state=state,
            public=public,
            auth=auth,
            inbox=inbox,
            debounce=debounce,
            storage=storage,
            version=version,
            lane=lane,
        )

        def decorator(handler: Handler[S]) -> A2AEndpoint[S]:
            ep = door(handler)
            self.add(ep)
            return ep

        return decorator

    def add(
        self, *doors: AgentHandler | RtcDoor | A2AEndpoint[Any]
    ) -> tuple[AgentHandler | RtcDoor | A2AEndpoint[Any], ...]:
        """Serve what a module defined: ``AgentHandler``s, and doors from ``rtc_session`` and
        ``a2a_session``, the one voice door of the worker and text doors under ``/<endpoint>``.
        Returns them."""
        from .a2a.auth import Credentials
        from .a2a.endpoint import A2AEndpoint

        for door in doors:
            if isinstance(door, AgentHandler):
                self.add(*door.doors)
            elif isinstance(door, RtcDoor):
                if self._entrypoint_fnc is not None:
                    raise RuntimeError("an AgentServer has one rtc_session")
                self._entrypoint_fnc = door.entrypoint
                self._request_fnc = door.on_request
                self._session_end_fnc = door.on_session_end
                self._simulation_end_fnc = door.on_simulation_end
                self._rtc_endpoint = door.endpoint or self._agent_name
                self._server_type = door.type
            elif isinstance(door, A2AEndpoint):
                if door.name in self._a2a_endpoints:
                    raise ValueError(f"a2a_session(endpoint={door.name!r}) is already served")
                credentials = (
                    Credentials(self._api_key, self._api_secret)
                    if self._api_key and self._api_secret
                    else None
                )
                door.bind(
                    storage=self._storage,
                    agent_name=lambda: self._agent_name,
                    credentials=credentials,
                    url_base=self._endpoint_url_base,
                    agents=self._agents,
                )
                if door.lane in (None, self._agent_name):
                    policy = door.turn_options.policy
                    person = policy.get("a2a") if isinstance(policy, dict) else policy
                    self._call_holds_lane &= person == "live"
                self._http.mount(f"/{door.name}", door.app, name=door.name)
                self._http.configure_endpoint(f"/{door.name}", public=door.public)
                self._a2a_endpoints[door.name] = door
            else:
                raise TypeError(f"not a door: {door!r}")
        return doors

    @property
    def a2a_endpoints(self) -> dict[str, A2AEndpoint[Any]]:
        """The text doors registered on this server, by endpoint name."""
        return dict(self._a2a_endpoints)

    def _endpoint_url_base(self) -> str | None:
        """``https://<project>.livekit.cloud/agents/<agent>/<deployment>`` when known (plan 7.2)."""
        if not self._ws_url or not self._agent_name:
            return None
        base = self._ws_url.rstrip("/")
        if base.startswith("wss://"):
            base = "https://" + base[len("wss://") :]
        elif base.startswith("ws://"):
            base = "http://" + base[len("ws://") :]
        # a worker registered without a deployment is addressed as "default" by the node
        return f"{base}/agents/{self._agent_name}/{self._deployment or 'default'}"

    def configure_endpoint(self, path: str, *, public: bool = False) -> None:
        """Set routing policy for one ``server.http`` route: ``public`` routes are reachable
        without a project token."""
        self._http.configure_endpoint(path, public=public)

    @property
    def log_level(self) -> str | ServerEnvOption[str]:
        return self._log_level

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
            session_end_timeout=options.session_end_timeout,
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
            prometheus_multiproc_dir=options.prometheus_multiproc_dir,
            setup_fnc=options.prewarm_fnc,
            load_fnc=options.load_fnc,
            log_level=options.log_level,
            agent_name=options.agent_name,
        )
        server.rtc_session(
            options.entrypoint_fnc,
            type=options.worker_type,
            on_request=options.request_fnc,
        )
        return server

    @overload
    def rtc_session(
        self,
        func: Callable[[JobContext], Awaitable[None]],
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> Callable[[JobContext], Awaitable[None]]: ...

    @overload
    def rtc_session(
        self,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> Callable[
        [Callable[[JobContext], Awaitable[None]]], Callable[[JobContext], Awaitable[None]]
    ]: ...

    def rtc_session(
        self,
        func: Callable[[JobContext], Awaitable[None]] | None = None,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> (
        Callable[[JobContext], Awaitable[None]]
        | Callable[
            [Callable[[JobContext], Awaitable[None]]], Callable[[JobContext], Awaitable[None]]
        ]
    ):
        """The voice door: the entrypoint a job runs. ``endpoint`` is the door's name under the
        agent, ``{agent_name}/{endpoint}``; the agent's name when left empty.
        ``livekit.agents.rtc_session`` is the same decorator with no server, for a door a
        module defines and ``add()`` serves.

        Usage:
            @server.rtc_session()
            async def my_agent(job_ctx: JobContext): ...

            server.rtc_session(my_agent)
        """

        def decorator(
            f: Callable[[JobContext], Awaitable[None]],
        ) -> Callable[[JobContext], Awaitable[None]]:
            self.add(
                RtcDoor(
                    f,
                    endpoint=endpoint,
                    type=type,
                    on_request=on_request,
                    on_session_end=on_session_end,
                    on_simulation_end=on_simulation_end,
                )
            )
            return f

        if func is not None:
            return decorator(func)

        return decorator

    @property
    def rtc_endpoint(self) -> str:
        """The voice door's name under the agent, ``{agent_name}/{endpoint}``."""
        return self._rtc_endpoint

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
        _core.init(_core.Role.Worker)

        async with self._lock:
            if not self._closed:
                raise Exception("worker is already running")

            if self._entrypoint_fnc is None:
                if not self._a2a_endpoints:
                    raise RuntimeError(
                        "No RTC session entrypoint has been registered.\n"
                        "Define one using the @server.rtc_session() decorator, for example:\n"
                        "    @server.rtc_session()\n"
                        "    async def my_agent(ctx: JobContext):\n"
                        "        ...\n"
                    )
                # a text-only server: an expert with nothing but a2a_session doors (plan 7.3)
                self._entrypoint_fnc = _text_only_entrypoint

            if self._request_fnc is None:
                self._request_fnc = _default_request_fnc

            if self._setup_fnc is None:
                self._setup_fnc = _default_setup_fnc

            if self.worker_info.cloud_agents:
                if self._load_fnc is not None:
                    logger.warning(
                        "custom load_fnc is not supported when hosting on Cloud, reverting to default"
                    )
                    self._load_fnc = None
                if self._load_threshold != _default_load_threshold:
                    logger.warning(
                        "custom load_threshold is not supported when hosting on Cloud, reverting to default"
                    )
                    self._load_threshold = _default_load_threshold

            if self._simulation:
                logger.info("simulation mode enabled: worker load limit disabled")

            self._loop = asyncio.get_event_loop()
            self._devmode = devmode
            self._job_lifecycle_tasks = set[asyncio.Task[Any]]()
            self._close_future: asyncio.Future[None] | None = None

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
                job_entrypoint_fnc=(
                    _CallThreadEntrypoint(
                        self._entrypoint_fnc, self._storage, self._call_holds_lane
                    )
                    if self._storage is not None
                    and self._entrypoint_fnc is not _text_only_entrypoint
                    else self._entrypoint_fnc
                ),
                session_end_fnc=self._session_end_fnc,
                simulation_end_fnc=self._simulation_end_fnc,
                num_idle_processes=ServerEnvOption.getvalue(self._num_idle_processes, devmode),
                loop=self._loop,
                job_executor_type=self._job_executor_type,
                inference_executor=self._inference_executor,
                mp_ctx=self._mp_ctx,
                initialize_timeout=self._initialize_process_timeout,
                close_timeout=self._shutdown_process_timeout,
                session_end_timeout=self._session_end_timeout,
                memory_warn_mb=self._job_memory_warn_mb,
                memory_limit_mb=self._job_memory_limit_mb,
                http_proxy=self._http_proxy or None,
            )

            self._api: api.LiveKitAPI | None = None
            self._http_session: aiohttp.ClientSession | None = None
            self._worker_load: float = 0.0

            # simulations run ephemeral workers side by side; a health
            # endpoint on a fixed port would make concurrent runs collide
            if not self._simulation:
                self._http_server = http_server.HttpServer(
                    self._host, ServerEnvOption.getvalue(self._port, devmode)
                )

                async def health_check(_: Any) -> web.Response:
                    if self._inference_executor and not self._inference_executor.is_alive():
                        return web.Response(status=503, text="inference process not running")

                    if self._connection_failed:
                        return web.Response(status=503, text="failed to connect to livekit")

                    return web.Response(text="OK")

                async def worker(_: Any) -> web.Response:
                    worker_info = agent_worker.WorkerInfo(
                        worker_type=agent.JobType.Name(self._server_type.value),
                        agent_name=self._agent_name,
                        active_jobs=len(self.active_jobs),
                        sdk_version=__version__,
                        worker_load=self._worker_load,
                        protocol_version=WORKER_PROTOCOL_VERSION,
                    )
                    body = MessageToJson(worker_info, preserving_proto_field_name=True)
                    return web.Response(body=body, content_type="application/json")

                self._http_server.app.add_routes([web.get("/", health_check)])
                self._http_server.app.add_routes([web.get("/worker", worker)])

            self._conn_task: asyncio.Task[None] | None = None
            self._load_task: asyncio.Task[None] | None = None

            if not unregistered:
                if not self._ws_url:
                    raise ValueError("ws_url is required, or set LIVEKIT_URL environment variable")

                if not self._api_key:
                    raise ValueError(
                        "api_key is required, or set LIVEKIT_API_KEY environment variable"
                    )

                if not self._api_secret:
                    raise ValueError(
                        "api_secret is required, or set LIVEKIT_API_SECRET environment variable"
                    )

                if self._http.manifest() and not self._agent_name:
                    raise ValueError(
                        "server.http routes are served at /agents/{agent_name}/..., "
                        "so they require an agent_name (AgentServer(agent_name=) or LIVEKIT_AGENT_NAME)"
                    )

            self._prometheus_server: telemetry.http_server.HttpServer | None = None
            if self._prometheus_port is not None:
                self._prometheus_server = telemetry.http_server.HttpServer(
                    self._host, self._prometheus_port
                )

            if self._prometheus_multiproc_dir:
                os.environ["PROMETHEUS_MULTIPROC_DIR"] = self._prometheus_multiproc_dir
            elif "PROMETHEUS_MULTIPROC_DIR" in os.environ:
                self._prometheus_multiproc_dir = os.environ["PROMETHEUS_MULTIPROC_DIR"]

            if self._prometheus_multiproc_dir:
                os.makedirs(self._prometheus_multiproc_dir, exist_ok=True)

            if self._prometheus_multiproc_dir and os.path.exists(self._prometheus_multiproc_dir):
                logger.debug(
                    "cleaning prometheus multiprocess directory",
                    extra={"path": self._prometheus_multiproc_dir},
                )
                for filename in os.listdir(self._prometheus_multiproc_dir):
                    file_path = os.path.join(self._prometheus_multiproc_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.warning(f"failed to remove {file_path}", exc_info=e)

            if self._ws_url:
                os.environ["LIVEKIT_URL"] = self._ws_url
            if self._api_key:
                os.environ["LIVEKIT_API_KEY"] = self._api_key
            if self._api_secret:
                os.environ["LIVEKIT_API_SECRET"] = self._api_secret
            if self._worker_token:
                os.environ["LIVEKIT_WORKER_TOKEN"] = self._worker_token
            if self._deployment:
                os.environ["LIVEKIT_AGENT_DEPLOYMENT"] = self._deployment

            # must run before job processes spawn so they inherit the SSL_CERT_FILE fallback
            http_context._set_default_cert_env()

            logger.info(
                "starting worker",
                extra={"version": __version__, "rtc-version": rtc.__version__},
            )

            for p in Plugin.registered_plugins:
                logger.info(
                    "plugin registered",
                    extra={
                        "plugin": p.title,
                        "version": p.version,
                    },
                )

            if self._mp_ctx_str == "forkserver":
                # the framework's warm-up runs once in the forkserver and is inherited via
                # COW; under `spawn` each job process imports it itself (see ipc._preload)
                plugin_packages = [p.package for p in Plugin.registered_plugins] + [
                    "livekit.agents.ipc._preload",
                    # Must remain last; it freezes objects imported by earlier preloads.
                    "livekit.agents.ipc._preload_freeze",
                ]
                logger.info("preloading plugins", extra={"packages": plugin_packages})
                self._mp_ctx.set_forkserver_preload(plugin_packages)

            if self._inference_executor is not None:
                logger.info("starting inference executor")
                await self._inference_executor.start()
                await self._inference_executor.initialize()

            self._closed = False

            def _update_job_status(proc: ipc.job_executor.JobExecutor) -> None:
                job_info = proc.running_job
                if job_info is None or self._conn is None:
                    return
                status = agent.JobStatus.JS_RUNNING
                if proc.status == ipc.job_executor.JobStatus.FAILED:
                    status = agent.JobStatus.JS_FAILED
                elif proc.status == ipc.job_executor.JobStatus.SUCCESS:
                    status = agent.JobStatus.JS_SUCCESS
                with contextlib.suppress(RuntimeError):  # the worker stopped
                    self._conn.job_status(job_info.job.id, int(status))

            if self._http_server is not None:
                await self._http_server.start()
                logger.info(
                    f"HTTP server listening on {self._http_server.host}:{self._http_server.port}"
                )

            if self._prometheus_server:
                await self._prometheus_server.start()
                logger.info(
                    "Prometheus metrics exposed at http://%s:%s/metrics",
                    self._prometheus_server.host,
                    self._prometheus_server.port,
                )

            self._proc_pool.on("process_started", _update_job_status)
            self._proc_pool.on("process_closed", _update_job_status)
            self._proc_pool.on("process_job_launched", _update_job_status)
            await self._proc_pool.start()

            self._http_session = aiohttp.ClientSession(
                proxy=self._http_proxy or None,
                connector=aiohttp.TCPConnector(ssl=http_context._create_ssl_context()),
            )
            if self._ws_url:
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

                    # the load and the idle-process target are the Rust worker's; an
                    # unregistered worker has neither
                    if self._conn is not None:
                        snapshot = self._conn.snapshot()
                        self._worker_load = snapshot["worker_load"]
                        # None leaves the pool unchanged; 0 means keep no idle process
                        if (target := snapshot["target_idle_processes"]) is not None:
                            self._proc_pool.set_target_idle_processes(target)

                    telemetry.metrics._update_worker_load(self._worker_load)
                    if self._prometheus_multiproc_dir:
                        await asyncio.get_event_loop().run_in_executor(
                            None, telemetry.metrics._update_child_proc_count
                        )

            tasks = []
            self._load_task = asyncio.create_task(_load_task(), name="load_task")
            tasks.append(self._load_task)

            if not unregistered:
                self._conn_task = asyncio.create_task(
                    self._connection_task(), name="worker_conn_task"
                )
                tasks.append(self._conn_task)

                # propagate a terminal connection failure out of run()
                def _on_conn_task_done(task: asyncio.Task[None]) -> None:
                    if task.cancelled() or self._close_future is None or self._close_future.done():
                        return
                    if (exc := task.exception()) is not None:
                        self._close_future.set_exception(exc)

                self._conn_task.add_done_callback(_on_conn_task_done)

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
        shutdown_process_timeout: NotGivenOr[float] = NOT_GIVEN,
        session_end_timeout: NotGivenOr[float] = NOT_GIVEN,
        initialize_process_timeout: NotGivenOr[float] = NOT_GIVEN,
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

        if is_given(session_end_timeout):
            self._session_end_timeout = session_end_timeout

        if is_given(initialize_process_timeout):
            self._initialize_process_timeout = initialize_process_timeout

    @property
    def id(self) -> str:
        return self._id

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return [proc.running_job for proc in self._proc_pool.processes if proc.running_job]

    @property
    def draining(self) -> bool:
        return self._draining

    async def drain(self, timeout: NotGivenOr[int | None] = NOT_GIVEN) -> None:
        """When timeout isn't None, it will raise asyncio.TimeoutError if the processes didn't finish in time."""  # noqa: E501

        timeout = timeout if is_given(timeout) else self._drain_timeout

        async with self._lock:
            if self._draining:
                return

            logger.info("draining worker", extra={"id": self.id, "timeout": timeout})
            self._draining = True
            # clients of an open WebSocket get a close frame and reconnect to another worker
            await self._http.close_websockets()
            if self._conn is not None:
                # Rust marks the worker full and waits for its pending requests and jobs
                await self._conn.drain(timeout or None)
                return

            async def _drain() -> None:
                # wait for in-flight availability tasks to finish launching their jobs
                await asyncio.gather(*self._job_lifecycle_tasks, return_exceptions=True)

                # then wait for the launched jobs to complete
                while True:
                    procs = [p for p in self._proc_pool.processes if p.running_job]
                    if not procs:
                        break
                    for proc in procs:
                        await proc.join()

            if timeout:
                await asyncio.wait_for(_drain(), timeout)  # raises asyncio.TimeoutError on timeout
            else:
                await _drain()

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

                room_info = models.Room(sid=utils.shortuuid("SRM_"), name=room)

            # room_info = await self._api.room.create_room(api.CreateRoomRequest(name=room))

            job = agent.Job(
                id=utils.shortuuid("job-") if not fake_job else utils.shortuuid("mock-job-"),
                room=room_info,
                type=agent.JobType.JT_ROOM,
                participant=None,
            )

            if not token:
                if fake_job and (not self._api_key or not self._api_secret):
                    token = ""
                else:
                    token = (
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
                if self._close_future is not None:
                    # _close_future may hold run()'s error; keep aclose() a no-op
                    with contextlib.suppress(Exception):
                        await self._close_future
                return

            self._closed = True

            # Worker never fully started (e.g. interrupted during init, or
            # running unregistered without a LiveKit connection)
            if self._close_future is None:
                return

            logger.info("shutting down worker", extra={"id": self.id})

            if self._conn is not None:
                with contextlib.suppress(RuntimeError):
                    self._conn.close()

            if self._conn_task is not None:
                await utils.aio.cancel_and_wait(self._conn_task)

            if self._load_task is not None:
                await utils.aio.cancel_and_wait(self._load_task)

            # let in-flight availability tasks finish launching their jobs
            # before closing the proc pool (they accepted before shutdown)
            await asyncio.gather(*self._job_lifecycle_tasks, return_exceptions=True)

            await self._proc_pool.aclose()

            if self._inference_executor is not None:
                await self._inference_executor.aclose()

            if self._http_session is not None:
                await self._http_session.close()

            if self._http_server is not None:
                await self._http_server.aclose()

            if self._prometheus_server:
                await self._prometheus_server.aclose()

            if self._api is not None:
                await self._api.aclose()  # type: ignore[no-untyped-call, unused-ignore]

            if not self._close_future.done():
                self._close_future.set_result(None)

    @utils.log_exceptions(logger=logger)
    async def _connection_task(self) -> None:
        manifest = self._http.manifest()
        options: dict[str, Any] = {
            "ws_url": self._ws_url,
            # the /agent address when it is not LIVEKIT_URL, such as the WebTransport port of a
            # development livekit-server (webtransport.port in its config); jobs still join rooms
            # at ws_url
            "worker_url": os.environ.get("LIVEKIT_WORKER_URL") or None,
            "agent_name": self._agent_name,
            "deployment": self._deployment,
            "version": __version__,
            "publisher": self._server_type == ServerType.PUBLISHER,
            "devmode": self._devmode,
            "simulation": self._simulation,
            "load_threshold": ServerEnvOption.getvalue(self._load_threshold, self._devmode),
            "num_idle_processes": ServerEnvOption.getvalue(self._num_idle_processes, self._devmode),
            "drain_timeout": self._drain_timeout,
            "worker_token": self._worker_token or None,
            "max_retry": self._max_retry,
            "permissions": dataclasses.asdict(self._permissions),
            "api_key": self._api_key,
            "api_secret": self._api_secret,
            "endpoints": manifest,
            # the endpoint data plane rides the WebTransport session; without routes the
            # WebSocket control connection suffices
            "webtransport": bool(manifest) or _env_flag("LIVEKIT_WORKER_WEBTRANSPORT"),
            "insecure": _env_flag("LIVEKIT_WORKER_INSECURE"),
        }
        conn = _core.Worker(options, _WorkerHost(self))
        self._conn = conn
        if manifest:
            logger.info(
                "serving agent endpoints",
                extra={"routes": [f"{','.join(e['methods'])} {e['path']}" for e in manifest]},
            )

        async def notifications() -> None:
            while True:
                event = await conn.next_notification()
                if event["kind"] == "registered":
                    self._id = event["worker_id"]
                    if self._storage is not None and not self._storage.holder:
                        self._storage.holder = self._id
                    self.emit(
                        "worker_registered",
                        self._id,
                        models.ServerInfo(region=event["region"], protocol=event["protocol"]),
                    )
                elif event["kind"] == "stopped":
                    return

        notification_task = asyncio.create_task(notifications(), name="worker_notifications")
        runtime_task = asyncio.ensure_future(conn.run())
        try:
            await asyncio.gather(runtime_task, notification_task)
        except Exception:
            self._connection_failed = True
            raise
        finally:
            with contextlib.suppress(RuntimeError):
                conn.close()
            await utils.aio.cancel_and_wait(runtime_task, notification_task)

    async def _invoke_load_fnc(self) -> float:
        """Run load_fnc in executor. Uses signature to call with or without self."""

        def load_fnc() -> float:
            assert self._load_fnc is not None
            signature = inspect.signature(self._load_fnc)
            parameters = list(signature.parameters.values())
            if len(parameters) == 0:
                return self._load_fnc()  # type: ignore
            return self._load_fnc(self)  # type: ignore

        return await asyncio.get_event_loop().run_in_executor(None, load_fnc)

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


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


class _WorkerHost:
    """Host operations the Rust worker opens, in the generator's ``open(kind, args)`` shape."""

    def __init__(self, server: AgentServer) -> None:
        self._server = server
        # (received_at, accepted_at) per accepted job, read back when the assignment launches it
        self._accept_times: dict[str, tuple[float, float]] = {}

    def open(self, kind: str, args: dict[str, Any]) -> Awaitable[Any]:
        if kind == "job_request":
            return self._job_request(args)
        if kind == "launch_job":
            return self._launch_job(args)
        if kind == "terminate_job":
            return self._terminate_job(args["job_id"])
        if kind == "load":
            return self._load()
        if kind == "endpoint":
            return self._server._http.serve(args)
        raise ValueError(f"unknown host operation {kind!r}")

    async def _job_request(self, req: dict[str, Any]) -> dict[str, Any]:
        server = self._server
        job = agent.Job()
        job.ParseFromString(req["job"]["proto"])
        received_at = time.time()
        answer: dict[str, Any] = {"decision": "unanswered"}

        async def _on_reject(terminate: bool) -> None:
            answer.clear()
            answer["decision"] = "terminate" if terminate else "reject"

        async def _on_accept(args: JobAcceptArguments) -> None:
            answer.clear()
            answer.update(
                decision="accept",
                name=args.name,
                identity=args.identity,
                metadata=args.metadata,
                attributes=list((args.attributes or {}).items()),
            )
            self._accept_times[job.id] = (received_at, time.time())

        # the core logs the request as it arrives (job, dispatch, room, resuming)
        job_req = JobRequest(job=job, on_reject=_on_reject, on_accept=_on_accept)
        assert server._request_fnc is not None
        try:
            await server._request_fnc(job_req)
        except Exception as e:
            logger.exception(
                "job_request_fnc failed",
                extra={"job_request": job_req, "agent_name": server._agent_name},
            )
            # `failed`, unlike `reject`, lets the server re-dispatch the job
            return {"decision": "failed", "error": f"{type(e).__name__}: {e}"}
        if answer["decision"] == "unanswered":
            logger.warning(
                "no answer was given inside the job_request_fnc, automatically rejecting the job",
                extra={"job_request": job_req, "agent_name": server._agent_name},
            )
        return answer

    async def _launch_job(self, info: dict[str, Any]) -> str:
        job = agent.Job()
        job.ParseFromString(info["job"]["proto"])
        a = info["accept_arguments"]
        received_at, accepted_at = self._accept_times.pop(job.id, (0.0, 0.0))
        running = RunningJobInfo(
            accept_arguments=JobAcceptArguments(
                name=a["name"],
                identity=a["identity"],
                metadata=a["metadata"],
                attributes=dict(a["attributes"]),
            ),
            job=job,
            url=info["url"],
            token=info["token"],
            worker_id=info["worker_id"],
            fake_job=info["fake_job"],
            received_at=received_at,
            accepted_at=accepted_at,
            assigned_at=time.time() if received_at else 0.0,
        )
        await self._server._proc_pool.launch_job(running)
        return job.id  # `job_status` is keyed on the LiveKit job id

    async def _terminate_job(self, job_id: str) -> None:
        proc = self._server._proc_pool.get_by_job_id(job_id)
        if proc:
            await proc.aclose()

    async def _load(self) -> float | None:
        """The user's load, or None to let Rust sample the CPU."""
        server = self._server
        if server._load_fnc is None:
            return None
        load = await server._invoke_load_fnc()
        server._worker_load = load
        telemetry.metrics._update_worker_load(load)
        return load
