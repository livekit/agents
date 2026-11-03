from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing as mp
import os
import threading
from abc import ABC
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

from typing_extensions import Unpack

from livekit import rtc
from livekit.protocol import agent, models

from . import _core, ipc, telemetry, utils
from ._generated.worker_options import ServerOptions
from ._generated.worker_server import (
    ProcessStartOptions,
    RunningJobInfo as CoreRunningJobInfo,
)
from .http import AgentRoutes
from .ipc.launch import preload_module_names, validate_setup_fnc
from .job import (
    JobAcceptArguments,
    JobContext,
    JobExecutorType,
    JobProcess,
    RunningJobInfo,
)
from .log import logger
from .simulation import SimulationContext
from .storage.context import _Handler
from .storage.resolve import _CallThreadEntrypoint
from .types import NOT_GIVEN, NotGivenOr
from .utils import http_context, is_given
from .version import __version__

if TYPE_CHECKING:
    from .a2a.endpoint import A2AEndpoint, A2ASessionOptions
    from .storage.backends import Storage


Entrypoint = Callable[[JobContext], Awaitable[None]]


_EntrypointT = TypeVar("_EntrypointT", bound=Entrypoint)


class RtcSessionOptions(TypedDict, total=False):
    """``rtc_session``'s parameters (docs/voice-session.md)."""

    endpoint: str
    """The door's name under the agent, ``{agent_name}/{endpoint}``; the agent's name when empty."""
    on_session_end: Callable[[JobContext], Any] | None
    on_simulation_end: Callable[[SimulationContext], Any] | None


class ListenerOptions(TypedDict, total=False):
    """Where a listener binds; a key left out keeps its default."""

    host: str
    port: int


class PublicListenerOptions(ListenerOptions, total=False):
    proxy_headers: bool
    """Read the client address and scheme from ``X-Forwarded-For`` and ``X-Forwarded-Proto``;
    only behind a proxy that always sets them."""


class Listeners(TypedDict, total=False):
    """``AgentServer(listeners=)``: public traffic and operational information never share a
    port (docs/server.md "Listeners")."""

    public: PublicListenerOptions
    """The agent's endpoints and HTTP routes, where the worker serves them (a self-hosted
    LiveKit server, the console); ``0.0.0.0:8080``."""
    health: ListenerOptions
    """``/``, ``/ready`` and ``/worker``; ``0.0.0.0:8081``."""
    prometheus: ListenerOptions
    """``/metrics``; off unless given, then ``0.0.0.0:9090``."""


@dataclass(frozen=True)
class RtcSession:
    """The voice door: what ``rtc_session`` returns and ``AgentServer.add`` serves."""

    entrypoint: Entrypoint
    endpoint: str = ""
    on_session_end: Callable[[JobContext], Any] | None = None
    on_simulation_end: Callable[[SimulationContext], Any] | None = None


def rtc_session(**options: Unpack[RtcSessionOptions]) -> Callable[[Entrypoint], RtcSession]:
    """The voice door as a decorator with no handler: the entrypoint a job runs, as an
    ``RtcSession`` that ``AgentServer.add()`` serves."""
    return lambda entrypoint: RtcSession(entrypoint, **options)


class AgentHandler:
    """An agent's doors on one lane: what a module defines and ``AgentServer.add`` serves.

    ::

        back_office = AgentHandler(lane="backend")

        @back_office.a2a_session(endpoint="backend", description="The back office.")
        async def text(ctx: A2ASessionContext) -> str: ...

    ``lane`` names the lane the doors write (docs/internals/storage.md#lanes); the worker's own when left out.
    The voice door writes the worker's lane, so a handler with a lane of its own has text
    doors only. ``AgentServer`` is the handler of the worker's own lane that serves its doors.
    """

    def __init__(self, *, lane: str | None = None) -> None:
        self.lane = lane
        self.doors: list[RtcSession | A2AEndpoint] = []

    def add(
        self, *doors: AgentHandler | RtcSession | A2AEndpoint
    ) -> tuple[AgentHandler | RtcSession | A2AEndpoint, ...]:
        for door in doors:
            self.doors.extend(door.doors if isinstance(door, AgentHandler) else [door])
        return doors

    def rtc_session(
        self, **options: Unpack[RtcSessionOptions]
    ) -> Callable[[_EntrypointT], _EntrypointT]:
        """The voice door, as ``livekit.agents.rtc_session``, served by this handler::

        @server.rtc_session()
        async def my_agent(ctx: JobContext): ...
        """
        if self.lane:
            raise ValueError(
                "the voice door writes the worker's lane: AgentHandler() without lane="
            )

        def decorator(entrypoint: _EntrypointT) -> _EntrypointT:
            self.add(RtcSession(entrypoint, **options))
            return entrypoint

        return decorator

    def a2a_session(
        self, *, endpoint: str, description: str, **options: Unpack[A2ASessionOptions]
    ) -> Callable[[_Handler], A2AEndpoint]:
        """The text door on this handler's lane, as ``livekit.agents.a2a_session``: one A2A
        endpoint served under ``/<endpoint>``::

            @server.a2a_session(endpoint="scheduling", description="Book or move an appointment.")
            async def scheduling(ctx: A2ASessionContext) -> str: ...
        """
        from .a2a.endpoint import A2AEndpoint

        def decorator(handler: _Handler) -> A2AEndpoint:
            ep = A2AEndpoint(
                handler, endpoint=endpoint, description=description, lane=self.lane, **options
            )
            self.add(ep)
            return ep

        return decorator


def _core_options(given: dict[str, Any]) -> ServerOptions:
    """The server options given by name, as the core takes them; the core has the defaults."""
    skip = (
        "self",
        "__class__",
        "storage",
        "project_id",
        "thread_api_port",
        "preload_modules",
        "setup_fnc",
        "_load_fnc",
    )
    options = {k: v for k, v in given.items() if k not in skip and is_given(v)}
    if "job_executor_type" in options:
        options["job_executor_type"] = options["job_executor_type"].value
    return ServerOptions(**options)


class AgentServer(AgentHandler):
    """The worker: its doors (one voice door, text doors under ``/<endpoint>``), the job
    processes, and the connection to LiveKit, whose lifecycle is the core's
    (``_core.ServerRuntime``). An option left out is the core's default, from the environment
    where one applies (``LIVEKIT_URL``, ``LIVEKIT_AGENT_NAME``, ...) and per mode otherwise
    (``load_threshold``, ``num_idle_processes`` and the listeners differ in development)."""

    def __init__(
        self,
        *,
        agent_name: NotGivenOr[str] = NOT_GIVEN,
        storage: Storage | None = None,
        project_id: str = "default",
        thread_api_port: int | None = None,
        ws_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        load_threshold: NotGivenOr[float] = NOT_GIVEN,
        job_memory_warn_mb: NotGivenOr[float] = NOT_GIVEN,
        job_memory_limit_mb: NotGivenOr[float] = NOT_GIVEN,
        drain_timeout: NotGivenOr[float] = NOT_GIVEN,
        shutdown_timeout: NotGivenOr[float] = NOT_GIVEN,
        num_idle_processes: NotGivenOr[int] = NOT_GIVEN,
        listeners: NotGivenOr[Listeners] = NOT_GIVEN,
        preload_modules: Sequence[str] = (),
        setup_fnc: Callable[[JobProcess], None] | None = None,
        _load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = None,
    ) -> None:
        options = _core_options({**locals(), "version": __version__})
        AgentHandler.__init__(self)
        self._config = _core.ServerConfig(options)
        self._storage = storage
        self._project_id = project_id
        self._thread_api_port = thread_api_port
        self._thread_server: _core._core_ThreadServer | None = None
        self._preload_modules = preload_module_names(preload_modules)
        self._setup_fnc = validate_setup_fnc(setup_fnc)
        self._load_fnc = _load_fnc
        self._on_started: list[Callable[[], Any]] = []
        self._rtc: RtcSession | None = None
        # HTTP routes served alongside the agent through the LiveKit server at
        # /agents/{agent_name}/{deployment}/{path}; the text doors are mounted on it
        self._http = AgentRoutes()
        self._a2a_endpoints: dict[str, A2AEndpoint] = {}
        self._runtime: _core.ServerRuntime | None = None
        self._jobs: _ServerCallbacks | None = None

    @property
    def storage(self) -> Storage | None:
        """The session storage every door of this server opens its lane with."""
        return self._storage

    def _ensure_storage(self) -> Storage:
        if self._storage is None:
            from .storage import SQLiteStorage

            self._storage = SQLiteStorage(project_id=self._project_id)
        return self._storage

    @property
    def thread_service_url(self) -> str | None:
        """Local ThreadService base URL while the development listener is running."""
        return f"http://127.0.0.1:{self._thread_server.port}" if self._thread_server else None

    @property
    def http(self) -> AgentRoutes:
        """HTTP routes served alongside the agent, as a Starlette application::

            @server.http.get("/token")
            async def token(request: Request) -> Response:
                return JSONResponse({"ok": True})

        Routes are served at ``/agents/{agent_name}/{deployment}/{path}``, through LiveKit Cloud
        or the public listener. The handler reads the request's preamble as
        ``request.state.lk_request``.
        """
        return self._http

    def add(
        self, *doors: AgentHandler | RtcSession | A2AEndpoint
    ) -> tuple[AgentHandler | RtcSession | A2AEndpoint, ...]:
        """Serve what a module defined: ``AgentHandler``s and the doors of ``rtc_session`` and
        ``a2a_session``, the one voice door of the worker and text doors under ``/<endpoint>``.
        Returns them."""
        from .a2a.endpoint import A2AEndpoint

        served = len(self.doors)
        super().add(*doors)
        for door in self.doors[served:]:
            if isinstance(door, RtcSession):
                if self._rtc is not None:
                    raise RuntimeError("an AgentServer has one rtc_session")
                self._rtc = door
                continue
            if not isinstance(door, A2AEndpoint):
                raise TypeError(f"not a door: {door!r}")
            if door.name in self._a2a_endpoints:
                raise ValueError(f"a2a_session(endpoint={door.name!r}) is already served")
            storage = self._ensure_storage()
            door.bind(storage=storage, config=self._config)
            self._http.mount(f"/{door.name}", door.app, name=door.name)
            self._http.configure_endpoint(f"/{door.name}", public=False)
            self._a2a_endpoints[door.name] = door
        return doors

    async def _run(self, *, devmode: bool = False, unregistered: bool = False) -> None:
        """Start the job processes and the health listener and, unless ``unregistered``, register
        with LiveKit; return once ``aclose()`` ran, or raise when the connection fails for good.

        Args:
            devmode: Development defaults (no idle processes, no load limit, debug logs).
            unregistered: Run without registering, invisible to the dispatcher (the console).
        """
        _core.init(_core.Role.Worker)
        from .a2a.remote import _referenced

        if unserved := sorted({e.name for e in _referenced if e._config is None}):
            logger.warning(
                "RemoteAgent refers to endpoints no AgentServer in this process serves; "
                "calls to them fail",
                extra={"endpoints": unserved},
            )
        storage = self._ensure_storage()
        self._jobs = _ServerCallbacks(self)
        self._runtime = _core.ServerRuntime(self._config, self._jobs)
        from .storage import SQLiteStorage

        try:
            if isinstance(storage, SQLiteStorage) and (
                (devmode and not unregistered) or self._thread_api_port is not None
            ):
                self._thread_server = await storage.threads._core_serve_threads(
                    self._config.api_key,
                    self._config.api_secret,
                    self._thread_api_port or 0,
                )
                logger.info(
                    "local thread service",
                    extra={
                        "url": self.thread_service_url,
                        "project_id": storage.threads.project_id,
                    },
                )
            await self._runtime.run(
                devmode=devmode,
                unregistered=unregistered,
                has_rtc=self._rtc is not None,
                has_text_doors=bool(self._a2a_endpoints),
                routes=self._http.manifest(),
                has_load_fnc=self._load_fnc is not None,
                endpoints=self._http._server,
            )
        finally:
            if self._thread_server is not None:
                await self._thread_server.aclose()
                self._thread_server = None

    def _update_options(self, **options: Any) -> None:
        """Change options of ``AgentServer()`` before the server starts."""
        self._config.update(_core_options(options))

    @property
    def id(self) -> str:
        return self._runtime.worker_id if self._runtime is not None else "unregistered"

    @property
    def _active_jobs(self) -> list[RunningJobInfo]:
        return self._jobs.active_jobs() if self._jobs is not None else []

    @property
    def draining(self) -> bool:
        return self._runtime is not None and self._runtime.draining

    async def _drain(self, timeout: NotGivenOr[float | None] = NOT_GIVEN) -> None:
        """Stop taking jobs and wait for the running ones; past ``timeout`` (the server's
        ``drain_timeout`` when not given) raises ``asyncio.TimeoutError``."""
        if self._runtime is not None:
            await self._runtime.drain(timeout if is_given(timeout) else self._config.drain_timeout)

    @utils.log_exceptions(logger=logger)
    async def _simulate_job(
        self,
        room: str,
        *,
        fake_job: bool = False,
        agent_identity: str | None = None,
        room_info: models.Room | None = None,
        token: str | None = None,
    ) -> None:
        """Run a job in ``room`` with no dispatch: the console's."""
        assert self._runtime is not None, "_simulate_job() needs a running server"
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


class _ServerCallbacks:
    """What the core's server runtime calls on the host: the job processes and the user's
    callbacks. Which jobs to take, when they were taken and whether job processes start at all
    are the core's decisions."""

    def __init__(self, server: AgentServer) -> None:
        self._server = server
        self._proc_pool: ipc.ProcPool | None = None

    def active_jobs(self) -> list[RunningJobInfo]:
        pool = self._proc_pool
        return [p.running_job for p in pool.processes if p.running_job] if pool else []

    async def launch(self, info: RunningJobInfo) -> None:
        assert self._proc_pool is not None
        await self._proc_pool.launch_job(info)

    # --- the lifecycle

    async def _core_start(self, opts: ProcessStartOptions) -> None:
        """The job processes, when the core wants them. The server's settings (``opts.env``)
        become a job process's environment; this process's is never changed."""
        # must run before job processes spawn so they inherit the SSL_CERT_FILE fallback
        http_context._set_default_cert_env()
        logger.info(
            "starting worker", extra={"version": __version__, "rtc-version": rtc.__version__}
        )
        for p in Plugin.registered_plugins:
            logger.info("plugin registered", extra={"plugin": p.title, "version": p.version})
        door = self._server._rtc
        if not opts.job_processes or door is None:
            return
        mp_ctx = mp.get_context(opts.multiprocessing_context)
        if opts.multiprocessing_context == "forkserver":
            # the framework's warm-up runs once in the forkserver and is inherited via COW;
            # under `spawn` each job process imports it itself (see ipc._preload)
            from .ipc.launch import configure_forkserver

            plugin_packages = [p.package for p in Plugin.registered_plugins]
            logger.info("preloading plugins", extra={"packages": plugin_packages})
            configure_forkserver(mp_ctx, [*plugin_packages, *self._server._preload_modules])
        loop = asyncio.get_running_loop()
        storage = self._server._storage
        self._proc_pool = ipc.ProcPool(
            job_entrypoint_fnc=(
                _CallThreadEntrypoint(door.entrypoint, storage) if storage else door.entrypoint
            ),
            session_end_fnc=door.on_session_end,
            simulation_end_fnc=door.on_simulation_end,
            preload_modules=self._server._preload_modules,
            setup_fnc=self._server._setup_fnc,
            loop=loop,
            job_executor_type=JobExecutorType(opts.job_executor_type),
            mp_ctx=mp_ctx,
            num_idle_processes=opts.pool.num_idle_processes,
            initialize_timeout=opts.pool.initialize_timeout,
            close_timeout=opts.pool.close_timeout,
            session_end_timeout=opts.pool.session_end_timeout,
            memory_warn_mb=opts.pool.memory_warn_mb,
            memory_limit_mb=opts.pool.memory_limit_mb,
            http_proxy=opts.pool.http_proxy,
            env=opts.env,
        )
        for event in ("process_started", "process_closed", "process_job_launched"):
            self._proc_pool.on(event, self._job_status)
        await self._proc_pool.start()

    async def _core_started(self) -> None:
        for callback in self._server._on_started:
            callback()

    async def _core_idle_processes(self, target: int) -> None:
        if self._proc_pool is not None:
            self._proc_pool.set_target_idle_processes(target)

    async def _core_worker_load(self, load: float) -> None:
        telemetry.metrics._update_worker_load(load)
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            await asyncio.get_running_loop().run_in_executor(
                None, telemetry.metrics._update_child_proc_count
            )

    async def _core_metrics(self) -> tuple[str, str]:
        return await asyncio.get_running_loop().run_in_executor(None, telemetry.metrics.exposition)

    async def _core_health(self) -> int:
        return len(self.active_jobs())

    async def _core_join_jobs(self) -> None:
        if self._proc_pool is not None:
            await self._proc_pool.join_jobs()

    async def _core_close(self) -> None:
        if self._proc_pool is not None:
            await self._proc_pool.aclose()

    def _job_status(self, proc: ipc.JobExecutor) -> None:
        if proc.running_job is not None and self._server._runtime is not None:
            self._server._runtime.job_status(proc.running_job.job.id, proc.status.value)

    # --- jobs and the user's load_fnc

    async def _core_launch_job(self, info: CoreRunningJobInfo) -> str:
        """A job the core took, with the timeline it stamped."""
        job = agent.Job.FromString(info.job.proto)
        await self.launch(
            RunningJobInfo(
                accept_arguments=JobAcceptArguments(
                    name=info.accept_arguments.name,
                    identity=info.accept_arguments.identity,
                    metadata=info.accept_arguments.metadata,
                    attributes=dict(info.accept_arguments.attributes),
                ),
                job=job,
                url=info.url,
                token=info.token,
                worker_id=info.worker_id,
                fake_job=info.fake_job,
                received_at=info.received_at,
                accepted_at=info.accepted_at,
                assigned_at=info.assigned_at,
            )
        )
        return job.id  # `job_status` is keyed on the LiveKit job id

    async def _core_terminate_job(self, job_id: str) -> None:
        if self._proc_pool is not None and (proc := self._proc_pool.get_by_job_id(job_id)):
            await proc.aclose()

    async def _core_load(self) -> float:
        """The user's ``load_fnc``, which the core asks only when there is one."""
        load_fnc: Callable[..., float] | None = self._server._load_fnc
        assert load_fnc is not None
        args = (self._server,) if inspect.signature(load_fnc).parameters else ()
        return await asyncio.get_running_loop().run_in_executor(None, load_fnc, *args)


PluginEventTypes = Literal["plugin_registered"]


class Plugin(ABC):  # noqa: B024
    registered_plugins: list[Plugin] = []
    emitter: utils.EventEmitter[PluginEventTypes] = utils.EventEmitter()

    # TODO(theomonnom): make logger mandatory once all plugins have been updated
    def __init__(
        self,
        title: str,
        version: str,
        package: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._title = title
        self._version = version
        self._package = package
        self._logger = logger

    @classmethod
    def register_plugin(cls, plugin: Plugin) -> None:
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Plugins must be registered on the main thread")

        cls.registered_plugins.append(plugin)
        cls.emitter.emit("plugin_registered", plugin)

    # plugin can implement an optional download_files method
    def download_files(self) -> None:  # noqa: B027
        pass

    @property
    def package(self) -> str:
        return self._package

    @property
    def title(self) -> str:
        return self._title

    @property
    def version(self) -> str:
        return self._version

    @property
    def logger(self) -> logging.Logger | None:
        return self._logger
