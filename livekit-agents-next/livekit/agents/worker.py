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
import multiprocessing as mp
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

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

# The server's lifecycle (options and their defaults, health listener, load, registration,
# availability, assignment, status, drain, close) and the endpoint data plane are the core's,
# `_core.ServerRuntime`; this module owns the doors, the job processes and `AgentServer.http`.


class ServerType(Enum):
    ROOM = agent.JobType.JT_ROOM
    PUBLISHER = agent.JobType.JT_PUBLISHER


@dataclass
class WorkerPermissions:
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
    can_update_metadata: bool = True
    can_publish_sources: list[models.TrackSource] = field(default_factory=list)
    hidden: bool = False


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


Entrypoint = Callable[[JobContext], Awaitable[None]]
_EntrypointT = TypeVar("_EntrypointT", bound=Entrypoint)


@dataclass(frozen=True)
class RtcDoor:
    """The voice door: what ``rtc_session`` returns and ``AgentServer.add`` serves."""

    entrypoint: Entrypoint
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
) -> Callable[[Entrypoint], RtcDoor]:
    """The voice door as a decorator with no handler: the entrypoint a job runs, as an
    ``RtcDoor`` that ``AgentServer.add()`` serves. ``endpoint`` is the door's name under the
    agent, ``{agent_name}/{endpoint}``; the agent's name when left empty."""
    return functools.partial(
        RtcDoor,
        endpoint=endpoint,
        type=type,
        on_request=on_request,
        on_session_end=on_session_end,
        on_simulation_end=on_simulation_end,
    )


class AgentHandler:
    """An agent's doors on one lane: what a module defines and ``AgentServer.add`` serves.

    ::

        back_office = AgentHandler(lane="backend")

        @back_office.a2a_session(endpoint="backend", description="The back office.", public=False)
        async def text(ctx: A2ASessionContext) -> str: ...

    ``lane`` names the lane the doors write (docs/lanes.md); the worker's own when left out.
    The voice door writes the worker's lane, so a handler with a lane of its own has text
    doors only. ``AgentServer`` is the handler of the worker's own lane that serves its doors.
    """

    def __init__(self, *, lane: str | None = None) -> None:
        self.lane = lane
        self.doors: list[RtcDoor | A2AEndpoint] = []

    def add(
        self, *doors: AgentHandler | RtcDoor | A2AEndpoint
    ) -> tuple[AgentHandler | RtcDoor | A2AEndpoint, ...]:
        for door in doors:
            self.doors.extend(door.doors if isinstance(door, AgentHandler) else [door])
        return doors

    @overload
    def rtc_session(
        self,
        func: _EntrypointT,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> _EntrypointT: ...

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
    ) -> Callable[[_EntrypointT], _EntrypointT]: ...

    def rtc_session(
        self,
        func: Entrypoint | None = None,
        *,
        endpoint: str = "",
        type: ServerType = ServerType.ROOM,
        on_request: Callable[[JobRequest], Any] | None = None,
        on_session_end: Callable[[JobContext], Any] | None = None,
        on_simulation_end: Callable[[SimulationContext], Any] | None = None,
    ) -> Any:
        """The voice door, as ``livekit.agents.rtc_session``, served by this handler::

        @server.rtc_session()
        async def my_agent(ctx: JobContext): ...
        """
        if self.lane:
            raise ValueError(
                "the voice door writes the worker's lane: AgentHandler() without lane="
            )
        door = rtc_session(
            endpoint=endpoint,
            type=type,
            on_request=on_request,
            on_session_end=on_session_end,
            on_simulation_end=on_simulation_end,
        )

        def decorator(f: Entrypoint) -> Entrypoint:
            self.add(door(f))
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
        """The text door on this handler's lane (docs/text-endpoints.md): one A2A endpoint
        served under ``/<endpoint>``, its card at ``/<endpoint>/.well-known/agent-card.json``
        and JSON-RPC over HTTP+SSE and WebSocket at ``/<endpoint>/v1``.

        ::

            @server.a2a_session(endpoint="scheduling", description="Book or move an appointment.")
            async def scheduling(ctx: A2ASessionContext) -> str: ...

        Args:
            endpoint: The last path segment of the endpoint's URL.
            description: What the card says the agent does.
            public: Whether a caller needs no token.
            auth: A callable deciding who may call.
            inbox: The turn policy for a message that arrives while one runs.
            debounce: How long to wait for more of a burst.
            storage: This endpoint's storage when it is not the server's.
            version: The card's version; the deployment's by default.
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
            lane=self.lane,
        )

        def decorator(handler: Handler) -> A2AEndpoint:
            ep = door(handler)
            self.add(ep)
            return ep

        return decorator


def _core_options(given: dict[str, Any]) -> dict[str, Any]:
    """The server options given by name, as the core takes them; the core has the defaults."""
    skip = ("self", "__class__", "storage", "setup_fnc", "load_fnc")
    options = {k: v for k, v in given.items() if k not in skip and is_given(v)}
    if "job_executor_type" in options:
        options["job_executor_type"] = options["job_executor_type"].value
    if "permissions" in options:
        options["permissions"] = dataclasses.asdict(options["permissions"])
    return options


class AgentServer(AgentHandler, utils.EventEmitter[Literal["worker_started", "worker_registered"]]):
    """The worker: its doors (one voice door, text doors under ``/<endpoint>``), the job
    processes, and the connection to LiveKit, whose lifecycle is the core's
    (``_core.ServerRuntime``). An option left out is the core's default, from the environment
    where one applies (``LIVEKIT_URL``, ``LIVEKIT_AGENT_NAME``, ...) and per mode otherwise
    (``load_threshold``, ``num_idle_processes`` and ``port`` differ in development)."""

    def __init__(
        self,
        *,
        agent_name: NotGivenOr[str] = NOT_GIVEN,
        storage: AgentDBStorage | None = None,
        setup_fnc: Callable[[JobProcess], Any] | None = None,
        load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = None,
        ws_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        job_executor_type: NotGivenOr[JobExecutorType] = NOT_GIVEN,
        load_threshold: NotGivenOr[float] = NOT_GIVEN,
        job_memory_warn_mb: NotGivenOr[float] = NOT_GIVEN,
        job_memory_limit_mb: NotGivenOr[float] = NOT_GIVEN,
        drain_timeout: NotGivenOr[float | None] = NOT_GIVEN,
        num_idle_processes: NotGivenOr[int] = NOT_GIVEN,
        shutdown_process_timeout: NotGivenOr[float] = NOT_GIVEN,
        session_end_timeout: NotGivenOr[float] = NOT_GIVEN,
        initialize_process_timeout: NotGivenOr[float] = NOT_GIVEN,
        permissions: NotGivenOr[WorkerPermissions] = NOT_GIVEN,
        max_retry: NotGivenOr[int] = NOT_GIVEN,
        host: NotGivenOr[str] = NOT_GIVEN,
        port: NotGivenOr[int] = NOT_GIVEN,
        http_proxy: NotGivenOr[str | None] = NOT_GIVEN,
        multiprocessing_context: NotGivenOr[Literal["spawn", "forkserver"]] = NOT_GIVEN,
        prometheus_port: NotGivenOr[int] = NOT_GIVEN,
        prometheus_multiproc_dir: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        options = _core_options(locals())
        AgentHandler.__init__(self)
        utils.EventEmitter.__init__(self)
        self._config = _core.ServerConfig({**options, "version": __version__})
        self._storage = storage
        self.setup_fnc = setup_fnc
        self.load_fnc = load_fnc
        self._rtc: RtcDoor | None = None
        # built here and not in run(): the @server.http decorators run at import time
        self._http = AgentHttp()
        self._a2a_endpoints: dict[str, A2AEndpoint] = {}
        self._runtime: _core.ServerRuntime | None = None
        self._jobs: _ServerHost | None = None

    @property
    def storage(self) -> AgentDBStorage | None:
        """The session storage every door of this server opens its lane with."""
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

    @property
    def a2a_endpoints(self) -> dict[str, A2AEndpoint]:
        """The text doors served, by endpoint name."""
        return dict(self._a2a_endpoints)

    def add(
        self, *doors: AgentHandler | RtcDoor | A2AEndpoint
    ) -> tuple[AgentHandler | RtcDoor | A2AEndpoint, ...]:
        """Serve what a module defined: ``AgentHandler``s and the doors of ``rtc_session`` and
        ``a2a_session``, the one voice door of the worker and text doors under ``/<endpoint>``.
        Returns them."""
        from .a2a.auth import Credentials
        from .a2a.endpoint import A2AEndpoint

        config = self._config
        served = len(self.doors)
        super().add(*doors)
        for door in self.doors[served:]:
            if isinstance(door, RtcDoor):
                if self._rtc is not None:
                    raise RuntimeError("an AgentServer has one rtc_session")
                self._rtc = door
                continue
            if not isinstance(door, A2AEndpoint):
                raise TypeError(f"not a door: {door!r}")
            if door.name in self._a2a_endpoints:
                raise ValueError(f"a2a_session(endpoint={door.name!r}) is already served")
            door.bind(
                storage=self._storage,
                agent_name=lambda: config.agent_name,
                credentials=(
                    Credentials(config.api_key, config.api_secret)
                    if config.api_key and config.api_secret
                    else None
                ),
                url_base=lambda: _core.endpoint_url_base(
                    config.ws_url, config.agent_name, config.deployment
                ),
            )
            self._http.mount(f"/{door.name}", door.app, name=door.name)
            self._http.configure_endpoint(f"/{door.name}", public=door.public)
            self._a2a_endpoints[door.name] = door
        return doors

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
        drain_timeout: NotGivenOr[float | None] = NOT_GIVEN,
        num_idle_processes: NotGivenOr[int] = NOT_GIVEN,
        shutdown_process_timeout: NotGivenOr[float] = NOT_GIVEN,
        session_end_timeout: NotGivenOr[float] = NOT_GIVEN,
        initialize_process_timeout: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Change the given options before the server starts."""
        self._config.update(_core_options(locals()))

    @property
    def id(self) -> str:
        return self._runtime.worker_id if self._runtime is not None else "unregistered"

    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        return self._jobs.active_jobs() if self._jobs is not None else []

    @property
    def draining(self) -> bool:
        return self._runtime is not None and self._runtime.draining

    async def drain(self, timeout: NotGivenOr[float | None] = NOT_GIVEN) -> None:
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
        call = functools.partial(load_fnc, self._server) if takes_server else load_fnc  # type: ignore[call-arg]
        return await asyncio.get_running_loop().run_in_executor(None, call)  # type: ignore[arg-type]
