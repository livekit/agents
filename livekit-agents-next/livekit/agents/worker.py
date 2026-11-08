from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

from typing_extensions import Unpack

from livekit import rtc
from livekit.protocol import models

from . import _core, ipc, telemetry, utils
from ._generated.job_records import RunningJobInfo
from ._generated.worker_options import ProcessOptions, ServerOptions, ServerOptionsKwargs
from ._generated.worker_server import ProcessStartOptions
from .evals import SimulationContext
from .http import AgentRoutes
from .ipc import preload_module_names
from .job import JobContext
from .log import logger
from .plugin import Plugin
from .storage import _CallThreadEntrypoint
from .storage.session import _Handler
from .types import NOT_GIVEN, NotGivenOr
from .utils import is_given
from .version import __version__

if TYPE_CHECKING:
    from .a2a import A2AEndpoint, A2ASessionOptions
    from .storage import Storage


Entrypoint = Callable[[JobContext], Awaitable[None]]


_EntrypointT = TypeVar("_EntrypointT", bound=Entrypoint)


class RtcSessionOptions(TypedDict, total=False):
    """``rtc_session``'s parameters."""

    endpoint: str
    """The door's name under the agent, ``{agent_name}/{endpoint}``; the agent's name when empty."""
    on_session_end: Callable[[JobContext], Any] | None
    on_simulation_end: Callable[[SimulationContext], Any] | None


@dataclass(frozen=True)
class RtcSession:
    """The voice door: what ``rtc_session`` returns and ``AgentServer.add`` serves."""

    entrypoint: Entrypoint
    endpoint: str = ""
    on_session_end: Callable[[JobContext], Any] | None = None
    on_simulation_end: Callable[[SimulationContext], Any] | None = None


def rtc_session(**options: Unpack[RtcSessionOptions]) -> Callable[[Entrypoint], RtcSession]:
    """Decorate a job entrypoint as an ``RtcSession`` for ``AgentServer.add()`` to serve."""
    return lambda entrypoint: RtcSession(entrypoint, **options)


class AgentHandler:
    """An agent's doors on one lane: what a module defines and ``AgentServer.add`` serves.

    ::

        back_office = AgentHandler(lane="backend")

        @back_office.a2a_session(endpoint="backend", description="The back office.")
        async def text(ctx: A2ASessionContext) -> str: ...

    ``lane`` names the lane the doors write; the worker's own when left out. The voice door
    writes the worker's lane, so a handler with a lane of its own has text doors only.
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
        """An A2A text door on this handler's lane, served under ``/<endpoint>``::

        @server.a2a_session(endpoint="scheduling", description="Book or move an appointment.")
        async def scheduling(ctx: A2ASessionContext) -> str: ...
        """
        from .a2a import A2AEndpoint

        def decorator(handler: _Handler) -> A2AEndpoint:
            ep = A2AEndpoint(
                handler, endpoint=endpoint, description=description, lane=self.lane, **options
            )
            self.add(ep)
            return ep

        return decorator


class AgentServer(AgentHandler):
    """The worker: its doors (one voice door, text doors under ``/<endpoint>``), the job
    processes, and the connection to LiveKit.

    Options left out default from the environment where one applies (``LIVEKIT_URL``,
    ``LIVEKIT_AGENT_NAME``, ...); ``load_threshold``, ``num_idle_processes`` and the listeners
    default differently in development.
    """

    def __init__(
        self,
        *,
        storage: Storage | None = None,
        project_id: str = "default",
        thread_api_port: int | None = None,
        preload_modules: Sequence[str] = (),
        _load_fnc: Callable[[AgentServer], float] | Callable[[], float] | None = None,
        **options: Unpack[ServerOptionsKwargs],
    ) -> None:
        AgentHandler.__init__(self)
        self._config = _core.ServerConfig(ServerOptions(**options), ProcessOptions(), __version__)
        self._storage = storage
        self._project_id = project_id
        self._thread_api_port = thread_api_port
        self._thread_server: _core._core_ThreadServer | None = None
        self._preload_modules = preload_module_names(preload_modules)
        self._load_fnc = _load_fnc
        self._on_started: list[Callable[[], Any]] = []
        self._rtc: RtcSession | None = None
        self._http = AgentRoutes()
        self._a2a_endpoints: dict[str, A2AEndpoint] = {}
        self._runtime: _core.ServerRuntime | None = None

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
        """HTTP and WebSocket routes served next to the agent; see ``AgentRoutes``."""
        return self._http

    def add(
        self, *doors: AgentHandler | RtcSession | A2AEndpoint
    ) -> tuple[AgentHandler | RtcSession | A2AEndpoint, ...]:
        """Serve ``AgentHandler``s and the doors of ``rtc_session`` and ``a2a_session``.

        At most one voice door per server. Returns ``doors``.
        """
        from .a2a import A2AEndpoint

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
            self._a2a_endpoints[door.name] = door
        return doors

    async def _run(self, *, devmode: bool = False, unregistered: bool = False) -> None:
        _core.init(_core.Role.Worker)
        telemetry._install()
        from .a2a import _referenced

        if unserved := sorted({e.name for e in _referenced if e._config is None}):
            logger.warning(
                "RemoteAgent refers to endpoints no AgentServer in this process serves; "
                "calls to them fail",
                extra={"endpoints": unserved},
            )
        storage = self._ensure_storage()
        runtime = self._ensure_runtime()
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
            await runtime.run(
                devmode=devmode,
                unregistered=unregistered,
                has_rtc=self._rtc is not None,
                has_text_doors=bool(self._a2a_endpoints),
                routes=self._http._manifest,
                has_load_fnc=self._load_fnc is not None,
                endpoints=self._http,
                text_doors=[door._door() for door in self._a2a_endpoints.values()],
                asyncio_debug=asyncio.get_running_loop().get_debug(),
            )
        finally:
            if self._thread_server is not None:
                await self._thread_server.aclose()
                self._thread_server = None
            await telemetry._shutdown()

    def _ensure_runtime(self) -> _core.ServerRuntime:
        if self._runtime is None:
            _core.init(_core.Role.Worker)
            self._runtime = _core.ServerRuntime(self._config, _ServerCallbacks(self))
        return self._runtime

    def _update_options(
        self, process: ProcessOptions | None = None, **options: Unpack[ServerOptionsKwargs]
    ) -> None:
        self._config.update(ServerOptions(**options), process or ProcessOptions())

    @property
    def id(self) -> str:
        return self._runtime.worker_id if self._runtime is not None else "unregistered"

    @property
    def _active_jobs(self) -> list[RunningJobInfo]:
        return self._runtime.active_jobs if self._runtime is not None else []

    @property
    def draining(self) -> bool:
        return self._runtime is not None and self._runtime.draining

    async def _drain(self, timeout: NotGivenOr[float | None] = NOT_GIVEN) -> None:
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
    def __init__(self, server: AgentServer) -> None:
        self._server = server

    async def _core_start(
        self, opts: ProcessStartOptions
    ) -> Callable[[], Awaitable[ipc.JobLauncher]] | None:
        # opts.env becomes the job processes' environment; this process's is never changed.
        logger.info(
            "starting worker", extra={"version": __version__, "rtc-version": rtc.__version__}
        )
        for p in Plugin.registered_plugins:
            logger.info("plugin registered", extra={"plugin": p.title, "version": p.version})
        door = self._server._rtc
        if not opts.job_processes or door is None:
            return None
        mp_ctx = mp.get_context(opts.multiprocessing_context)
        if opts.multiprocessing_context == "forkserver":
            # Warm-up runs once in the forkserver and is inherited via COW; under spawn each
            # job process imports it itself.
            from .ipc import configure_forkserver

            plugin_packages = [p.package for p in Plugin.registered_plugins]
            logger.info("preloading plugins", extra={"packages": plugin_packages})
            configure_forkserver(mp_ctx, [*plugin_packages, *self._server._preload_modules])
        loop = asyncio.get_running_loop()
        storage = self._server._storage
        entrypoint = _CallThreadEntrypoint(door.entrypoint, storage) if storage else door.entrypoint

        async def launcher() -> ipc.JobLauncher:
            return ipc.JobLauncher(
                job_entrypoint_fnc=entrypoint,
                session_end_fnc=door.on_session_end,
                simulation_end_fnc=door.on_simulation_end,
                session_end_timeout=opts.session_end_timeout,
                preload_modules=self._server._preload_modules,
                env=opts.env,
                mp_ctx=mp_ctx,
                loop=loop,
            )

        return launcher

    async def _core_started(self) -> None:
        for callback in self._server._on_started:
            callback()

    async def _core_heartbeat(self) -> None:
        pass

    async def _core_load(self) -> float:
        load_fnc: Callable[..., float] | None = self._server._load_fnc
        assert load_fnc is not None
        args = (self._server,) if inspect.signature(load_fnc).parameters else ()
        return await asyncio.get_running_loop().run_in_executor(None, load_fnc, *args)
