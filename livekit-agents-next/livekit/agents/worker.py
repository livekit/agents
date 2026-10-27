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
import dataclasses
import datetime
import functools
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

from starlette.types import ASGIApp

from livekit import rtc
from livekit.protocol import agent, models

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
from .storage.context import Handler
from .storage.protocol import StorageError
from .storage.resolve import Thread, resolve_thread
from .types import NOT_GIVEN, NotGivenOr
from .utils import http_context, is_given
from .version import __version__

if TYPE_CHECKING:
    from .a2a.endpoint import A2AEndpoint, AuthCallable

# The server's lifecycle (options, health listener, load, registration, availability, assignment,
# status, drain, close) and the endpoint data plane are the core's, `_core.ServerRuntime`; this
# module owns the doors, the job processes and `AgentServer.http`.
DRAIN_TIMEOUT = 3600  # 1hr


@dataclass(frozen=True)
class _CallThreadEntrypoint:
    """The ``rtc_session`` entrypoint of a server with storage: the call's thread is resolved
    first, the ``thread_id`` of the dispatch metadata, else a new thread, as ``ctx.thread``.
    A storage failure is logged and the call runs with no thread."""

    entrypoint: Callable[[JobContext], Awaitable[None]]
    storage: AgentDBStorage

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
            ctx._thread = Thread(self.storage, thread_id, agent_name=ctx.job.agent_name)
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
_default_num_idle_processes = ServerEnvOption(
    dev_default=0, prod_default=math.ceil(_core.cpu_count())
)
_default_port = ServerEnvOption(dev_default=0, prod_default=8081)


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
        level = _core.normalize_log_level(self.log_level)
        self.log_level = ServerEnvOption(*level) if isinstance(level, tuple) else level

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
        self.doors: list[RtcDoor | A2AEndpoint] = []

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
        public: bool = True,
        auth: AuthCallable | None = None,
        inbox: Policy | Mapping[str, Policy] | None = None,
        debounce: datetime.timedelta | Mapping[str, datetime.timedelta] | None = None,
        storage: AgentDBStorage | None = None,
        version: str | None = None,
    ) -> Callable[[Handler], A2AEndpoint]:
        """The text door, as ``AgentServer.a2a_session``, on this handler's lane."""
        from .a2a.endpoint import a2a_session

        door = a2a_session(
            endpoint=endpoint,
            description=description,
            public=public,
            auth=auth,
            inbox=inbox,
            debounce=debounce,
            storage=storage,
            version=version,
            lane=self.lane,
        )

        def decorator(handler: Handler) -> A2AEndpoint:
            ep = door(handler)
            self.doors.append(ep)
            return ep

        return decorator


class AgentServer(utils.EventEmitter[EventTypes]):
    """The worker: its doors (one voice door, text doors under ``/<endpoint>``), the job
    processes, and the connection to LiveKit, whose lifecycle is the core's
    (``_core.ServerRuntime``)."""

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
        options: dict[str, Any] = {
            "job_executor_type": job_executor_type.value,
            "load_threshold": load_threshold,
            "job_memory_warn_mb": job_memory_warn_mb,
            "job_memory_limit_mb": job_memory_limit_mb,
            "drain_timeout": drain_timeout,
            "num_idle_processes": num_idle_processes,
            "shutdown_process_timeout": shutdown_process_timeout,
            "session_end_timeout": session_end_timeout,
            "initialize_process_timeout": initialize_process_timeout,
            "permissions": dataclasses.asdict(permissions),
            "max_retry": max_retry,
            "ws_url": ws_url,
            "api_key": api_key,
            "api_secret": api_secret,
            "host": host,
            "port": port,
            "multiprocessing_context": multiprocessing_context,
            "prometheus_port": prometheus_port,
            "prometheus_multiproc_dir": prometheus_multiproc_dir,
            "log_level": log_level,
            "agent_name": agent_name,
            "version": __version__,
        }
        if is_given(http_proxy):
            options["http_proxy"] = http_proxy
        # the options, resolved against the environment by the core
        self._config = _core.ServerConfig(options)
        # the session storage every door of this server opens its lane with (plan 5.7)
        self._storage = storage
        self._setup_fnc = setup_fnc
        self._load_fnc = load_fnc
        self._rtc: RtcDoor | None = None
        # built here and not in run(): the @server.http decorators run at import time
        self._http = AgentHttp()
        self._a2a_endpoints: dict[str, A2AEndpoint] = {}
        self._runtime: _core.ServerRuntime | None = None
        self._jobs: _ServerHost | None = None

    @property
    def storage(self) -> AgentDBStorage | None:
        """The session storage every door of this server opens its lane with (plan 5.7)."""
        return self._storage

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
        public: bool = True,
        auth: AuthCallable | None = None,
        inbox: Policy | Mapping[str, Policy] | None = None,
        debounce: datetime.timedelta | Mapping[str, datetime.timedelta] | None = None,
        storage: AgentDBStorage | None = None,
        version: str | None = None,
        lane: str | None = None,
    ) -> Callable[[Handler], A2AEndpoint]:
        """The text door (plan 7.3): one A2A endpoint served as an ASGI app under ``/<endpoint>``.

        ::

            @server.a2a_session(endpoint="scheduling", description="Book or move an appointment.")
            async def scheduling(ctx: A2ASessionContext) -> str:
                result = await scheduler.run(ctx.message, message_history=ctx.message_history, context=app)
                return str(result.output)

        The endpoint publishes its card at ``/<endpoint>/.well-known/agent-card.json`` and
        serves JSON-RPC over HTTP+SSE and over WebSocket at ``/<endpoint>/v1``. The route is
        a session route: the node resolves the thread and hands it over in the preamble.
        ``livekit.agents.a2a_session`` is the same decorator with no server, for a door a
        module defines and ``add()`` serves.

        Args:
            endpoint: The last path segment of the endpoint's URL.
            description: What the card says the agent does.
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
            public=public,
            auth=auth,
            inbox=inbox,
            debounce=debounce,
            storage=storage,
            version=version,
            lane=lane,
        )

        def decorator(handler: Handler) -> A2AEndpoint:
            ep = door(handler)
            self.add(ep)
            return ep

        return decorator

    def add(
        self, *doors: AgentHandler | RtcDoor | A2AEndpoint
    ) -> tuple[AgentHandler | RtcDoor | A2AEndpoint, ...]:
        """Serve what a module defined: ``AgentHandler``s, and doors from ``rtc_session`` and
        ``a2a_session``, the one voice door of the worker and text doors under ``/<endpoint>``.
        Returns them."""
        from .a2a.auth import Credentials
        from .a2a.endpoint import A2AEndpoint

        for door in doors:
            if isinstance(door, AgentHandler):
                self.add(*door.doors)
            elif isinstance(door, RtcDoor):
                if self._rtc is not None:
                    raise RuntimeError("an AgentServer has one rtc_session")
                self._rtc = door
            elif isinstance(door, A2AEndpoint):
                if door.name in self._a2a_endpoints:
                    raise ValueError(f"a2a_session(endpoint={door.name!r}) is already served")
                config = self._config
                credentials = (
                    Credentials(config.api_key, config.api_secret)
                    if config.api_key and config.api_secret
                    else None
                )
                door.bind(
                    storage=self._storage,
                    agent_name=lambda: self._config.agent_name,
                    credentials=credentials,
                    url_base=self._endpoint_url_base,
                )
                self._http.mount(f"/{door.name}", door.app, name=door.name)
                self._http.configure_endpoint(f"/{door.name}", public=door.public)
                self._a2a_endpoints[door.name] = door
            else:
                raise TypeError(f"not a door: {door!r}")
        return doors

    @property
    def a2a_endpoints(self) -> dict[str, A2AEndpoint]:
        """The text doors registered on this server, by endpoint name."""
        return dict(self._a2a_endpoints)

    def _endpoint_url_base(self) -> str | None:
        """``https://<project>.livekit.cloud/agents/<agent>/<deployment>`` when known (plan 7.2)."""
        return _core.endpoint_url_base(  # type: ignore[no-any-return]
            self._config.ws_url, self._config.agent_name, self._config.deployment
        )

    def configure_endpoint(self, path: str, *, public: bool = False) -> None:
        """Set routing policy for one ``server.http`` route: ``public`` routes are reachable
        without a project token."""
        self._http.configure_endpoint(path, public=public)

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

    @property
    def log_level(self) -> str | ServerEnvOption[str]:
        level = self._config.log_level
        return ServerEnvOption(*level) if isinstance(level, tuple) else level

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
        if self._rtc is None:
            return ""
        return self._rtc.endpoint or self._config.agent_name

    @property
    def worker_info(self) -> WorkerInfo:
        port = self._runtime.http_port if self._runtime is not None else 0
        return WorkerInfo(http_port=port, cloud_agents=self._config.cloud)

    async def run(self, *, devmode: bool = False, unregistered: bool = False) -> None:
        """Start the job processes and the health listener and, unless ``unregistered``, register
        with LiveKit; return once ``aclose()`` ran, or raise when the connection fails for good.

        Args:
            devmode: Development defaults (no idle processes, no load limit, debug logs).
            unregistered: Run without registering, invisible to the dispatcher (the console).
        """
        _core.init(_core.Role.Worker)
        self._jobs = _ServerHost(self)
        self._runtime = _core.ServerRuntime(self._config, self._jobs)
        await self._runtime.run(
            devmode=devmode,
            unregistered=unregistered,
            has_rtc=self._rtc is not None,
            has_text_doors=bool(self._a2a_endpoints),
            publisher=self._rtc is not None and self._rtc.type == ServerType.PUBLISHER,
            routes=self._http.manifest(),
            has_load_fnc=self.load_fnc is not None,
        )

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
        given = {k: v for k, v in locals().items() if k != "self" and is_given(v)}
        if is_given(job_executor_type):
            given["job_executor_type"] = job_executor_type.value
        self._config.update(given)

    @property
    def id(self) -> str:
        return self._runtime.worker_id if self._runtime is not None else "unregistered"

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return self._jobs.active_jobs() if self._jobs is not None else []

    @property
    def draining(self) -> bool:
        return self._runtime is not None and self._runtime.draining

    async def drain(self, timeout: NotGivenOr[int | None] = NOT_GIVEN) -> None:
        """Stop taking jobs and wait for the running ones; past ``timeout`` (the server's
        ``drain_timeout`` when not given) raises ``asyncio.TimeoutError``."""
        if self._runtime is not None:
            await self._runtime.drain(timeout if is_given(timeout) else self._config.drain_timeout)

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
        """Run a job in ``room`` with no dispatch: the console's, a notebook's."""
        assert self._runtime is not None, "simulate_job() needs a running server"
        await self._runtime.simulate_job(
            room,
            fake_job=fake_job,
            agent_identity=agent_identity,
            room_info=room_info.SerializeToString() if room_info is not None else None,
            token=token,
        )

    async def aclose(self) -> None:
        if self._runtime is not None:
            await self._runtime.aclose()

    async def _reload_jobs(self, jobs: list[RunningJobInfo]) -> None:
        """Relaunch the jobs a dev reload carried over, each on a fresh token."""
        config = self._config
        for aj in jobs:
            logger.log(
                DEV_LEVEL,
                "reloading job",
                extra={"job_id": aj.job.id, "agent_name": aj.job.agent_name},
            )
            token = _core.extend_job_token(aj.token, config.api_key, config.api_secret)
            assert self._jobs is not None
            await self._jobs.launch(dataclasses.replace(aj, url=config.ws_url, token=token))


class _ServerHost:
    """What the core's server runtime opens on the host (``open(kind, args)``): the user's
    callbacks and the job processes, which are per language."""

    def __init__(self, server: AgentServer) -> None:
        self._server = server
        self._proc_pool: ipc.proc_pool.ProcPool | None = None
        self._inference_executor: ipc.inference_proc_executor.InferenceProcExecutor | None = None
        self._prometheus_server: telemetry.http_server.HttpServer | None = None
        # (received_at, accepted_at) per accepted job, read back when the assignment launches it
        self._accept_times: dict[str, tuple[float, float]] = {}

    def open(self, kind: str, args: Any) -> Awaitable[Any]:
        if kind == "endpoint":
            return self._server._http.serve(args)
        return getattr(self, f"_{kind}")(args)  # type: ignore[no-any-return]

    def active_jobs(self) -> list[RunningJobInfo]:
        pool = self._proc_pool
        return [p.running_job for p in pool.processes if p.running_job] if pool else []

    async def launch(self, info: RunningJobInfo) -> None:
        assert self._proc_pool is not None
        await self._proc_pool.launch_job(info)

    # --- the lifecycle

    async def _start(self, opts: dict[str, Any]) -> None:
        """The job processes, from the options the core resolved."""
        server = self._server
        os.environ.update(opts["env"])
        # must run before job processes spawn so they inherit the SSL_CERT_FILE fallback
        http_context._set_default_cert_env()
        logger.info(
            "starting worker", extra={"version": __version__, "rtc-version": rtc.__version__}
        )
        for p in Plugin.registered_plugins:
            logger.info("plugin registered", extra={"plugin": p.title, "version": p.version})
        mp_ctx = mp.get_context(opts["multiprocessing_context"])
        if opts["multiprocessing_context"] == "forkserver":
            # the framework's warm-up runs once in the forkserver and is inherited via COW;
            # under `spawn` each job process imports it itself (see ipc._preload)
            plugin_packages = [p.package for p in Plugin.registered_plugins] + [
                "livekit.agents.ipc._preload",
                # Must remain last; it freezes objects imported by earlier preloads.
                "livekit.agents.ipc._preload_freeze",
            ]
            logger.info("preloading plugins", extra={"packages": plugin_packages})
            mp_ctx.set_forkserver_preload(plugin_packages)
        loop = asyncio.get_running_loop()
        if _InferenceRunner.registered_runners:
            self._inference_executor = ipc.inference_proc_executor.InferenceProcExecutor(
                runners=_InferenceRunner.registered_runners,
                initialize_timeout=5 * 60,
                close_timeout=5,
                memory_warn_mb=2000,
                memory_limit_mb=0,  # no limit
                ping_interval=5,
                ping_timeout=60,
                high_ping_threshold=2.5,
                mp_ctx=mp_ctx,
                loop=loop,
                http_proxy=opts["http_proxy"] or None,
            )
            logger.info("starting inference executor")
            await self._inference_executor.start()
            await self._inference_executor.initialize()
        rtc_door = server._rtc
        entrypoint = rtc_door.entrypoint if rtc_door is not None else _text_only_entrypoint
        if server._storage is not None and rtc_door is not None:
            entrypoint = _CallThreadEntrypoint(entrypoint, server._storage)
        self._proc_pool = ipc.proc_pool.ProcPool(
            initialize_process_fnc=server.setup_fnc or _default_setup_fnc,
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=rtc_door.on_session_end if rtc_door else None,
            simulation_end_fnc=rtc_door.on_simulation_end if rtc_door else None,
            num_idle_processes=opts["num_idle_processes"],
            loop=loop,
            job_executor_type=JobExecutorType(opts["job_executor_type"]),
            inference_executor=self._inference_executor,
            mp_ctx=mp_ctx,
            initialize_timeout=opts["initialize_process_timeout"],
            close_timeout=opts["shutdown_process_timeout"],
            session_end_timeout=opts["session_end_timeout"],
            memory_warn_mb=opts["job_memory_warn_mb"],
            memory_limit_mb=opts["job_memory_limit_mb"],
            http_proxy=opts["http_proxy"] or None,
        )
        if opts["prometheus_port"] is not None:
            self._prometheus_server = telemetry.http_server.HttpServer(
                opts["host"], opts["prometheus_port"]
            )
            await self._prometheus_server.start()
            logger.info(
                "Prometheus metrics exposed at http://%s:%s/metrics",
                self._prometheus_server.host,
                self._prometheus_server.port,
            )
        for event in ("process_started", "process_closed", "process_job_launched"):
            self._proc_pool.on(event, self._job_status)
        await self._proc_pool.start()

    async def _started(self, _: Any) -> None:
        self._server.emit("worker_started")

    async def _registered(self, info: dict[str, Any]) -> None:
        server = self._server
        if server._storage is not None and not server._storage.holder:
            server._storage.holder = info["worker_id"]
        server.emit(
            "worker_registered",
            info["worker_id"],
            models.ServerInfo(region=info["region"], protocol=info["protocol"]),
        )

    async def _idle_processes(self, target: int) -> None:
        if self._proc_pool is not None:
            self._proc_pool.set_target_idle_processes(target)

    async def _worker_load(self, load: float) -> None:
        telemetry.metrics._update_worker_load(load)
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            await asyncio.get_running_loop().run_in_executor(
                None, telemetry.metrics._update_child_proc_count
            )

    async def _health(self, _: Any) -> tuple[bool, int]:
        alive = self._inference_executor is None or self._inference_executor.is_alive()
        return alive, len(self.active_jobs())

    async def _close_websockets(self, _: Any) -> None:
        # clients of an open WebSocket get a close frame and reconnect to another worker
        await self._server._http.close_websockets()

    async def _join_jobs(self, _: Any) -> None:
        while self._proc_pool and (
            procs := [p for p in self._proc_pool.processes if p.running_job]
        ):
            await asyncio.gather(*(proc.join() for proc in procs))
            # a joined process leaves the pool on the next loop turn
            await asyncio.sleep(0)

    async def _close(self, _: Any) -> None:
        if self._proc_pool is not None:
            await self._proc_pool.aclose()
        if self._inference_executor is not None:
            await self._inference_executor.aclose()
        if self._prometheus_server is not None:
            await self._prometheus_server.aclose()

    def _job_status(self, proc: ipc.job_executor.JobExecutor) -> None:
        if proc.running_job is not None and self._server._runtime is not None:
            self._server._runtime.job_status(proc.running_job.job.id, proc.status.value)

    # --- the worker's callbacks

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
        request_fnc = server._rtc.on_request if server._rtc else None
        try:
            await (request_fnc or _default_request_fnc)(job_req)
        except Exception as e:
            logger.exception(
                "job_request_fnc failed",
                extra={"job_request": job_req, "agent_name": server._config.agent_name},
            )
            # `failed`, unlike `reject`, lets the server re-dispatch the job
            return {"decision": "failed", "error": f"{type(e).__name__}: {e}"}
        if answer["decision"] == "unanswered":
            logger.warning(
                "no answer was given inside the job_request_fnc, automatically rejecting the job",
                extra={"job_request": job_req, "agent_name": server._config.agent_name},
            )
        return answer

    async def _launch_job(self, info: dict[str, Any]) -> str:
        job = agent.Job()
        job.ParseFromString(info["job"]["proto"])
        a = info["accept_arguments"]
        received_at, accepted_at = self._accept_times.pop(job.id, (0.0, 0.0))
        await self.launch(
            RunningJobInfo(
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
        )
        return job.id  # `job_status` is keyed on the LiveKit job id

    async def _terminate_job(self, args: dict[str, Any]) -> None:
        if self._proc_pool is not None and (proc := self._proc_pool.get_by_job_id(args["job_id"])):
            await proc.aclose()

    async def _load(self, _: Any) -> float | None:
        """The user's load, or None to let the core sample the CPU."""
        load_fnc = self._server.load_fnc
        if load_fnc is None:
            return None
        takes_server = bool(inspect.signature(load_fnc).parameters)
        call = functools.partial(load_fnc, self._server) if takes_server else load_fnc
        return await asyncio.get_running_loop().run_in_executor(None, call)  # type: ignore[arg-type]
