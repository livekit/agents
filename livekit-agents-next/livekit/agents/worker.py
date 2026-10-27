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
import inspect
import multiprocessing as mp
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

from typing_extensions import Unpack

from livekit import rtc
from livekit.protocol import agent, models

from . import _core, ipc, telemetry, utils
from .http import AgentHttp
from .inference_runner import _InferenceRunner
from .job import (
    JobAcceptArguments,
    JobContext,
    JobExecutorType,
    JobRequest,
    RunningJobInfo,
)
from .log import logger
from .plugin import Plugin
from .simulation import SimulationContext
from .storage.agentdb import AgentDBStorage
from .storage.context import Handler
from .storage.resolve import CallThreadEntrypoint
from .types import NOT_GIVEN, NotGivenOr
from .utils import http_context, is_given
from .version import __version__

if TYPE_CHECKING:
    from .a2a.endpoint import A2AEndpoint, A2ASessionOptions

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


Entrypoint = Callable[[JobContext], Awaitable[None]]
_EntrypointT = TypeVar("_EntrypointT", bound=Entrypoint)


class RtcSessionOptions(TypedDict, total=False):
    """``rtc_session``'s parameters (docs/voice.md)."""

    endpoint: str
    """The door's name under the agent, ``{agent_name}/{endpoint}``; the agent's name when empty."""
    type: ServerType
    """A job per room (the default) or per publisher."""
    on_request: Callable[[JobRequest], Any] | None
    """Decides on each job request; every job is accepted without one."""
    on_session_end: Callable[[JobContext], Any] | None
    on_simulation_end: Callable[[SimulationContext], Any] | None


@dataclass(frozen=True)
class RtcDoor:
    """The voice door: what ``rtc_session`` returns and ``AgentServer.add`` serves."""

    entrypoint: Entrypoint
    endpoint: str = ""
    type: ServerType = ServerType.ROOM
    on_request: Callable[[JobRequest], Any] | None = None
    on_session_end: Callable[[JobContext], Any] | None = None
    on_simulation_end: Callable[[SimulationContext], Any] | None = None


def rtc_session(**options: Unpack[RtcSessionOptions]) -> Callable[[Entrypoint], RtcDoor]:
    """The voice door as a decorator with no handler: the entrypoint a job runs, as an
    ``RtcDoor`` that ``AgentServer.add()`` serves."""
    return lambda entrypoint: RtcDoor(entrypoint, **options)


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
            self.add(RtcDoor(entrypoint, **options))
            return entrypoint

        return decorator

    def a2a_session(
        self, *, endpoint: str, description: str, **options: Unpack[A2ASessionOptions]
    ) -> Callable[[Handler], A2AEndpoint]:
        """The text door on this handler's lane, as ``livekit.agents.a2a_session``: one A2A
        endpoint served under ``/<endpoint>``::

            @server.a2a_session(endpoint="scheduling", description="Book or move an appointment.")
            async def scheduling(ctx: A2ASessionContext) -> str: ...
        """
        from .a2a.endpoint import a2a_session

        door = a2a_session(endpoint=endpoint, description=description, lane=self.lane, **options)

        def decorator(handler: Handler) -> A2AEndpoint:
            ep = door(handler)
            self.add(ep)
            return ep

        return decorator


def _core_options(given: dict[str, Any]) -> dict[str, Any]:
    """The server options given by name, as the core takes them; the core has the defaults."""
    skip = ("self", "__class__", "storage", "load_fnc")
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
        from .a2a.endpoint import A2AEndpoint

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
            door.bind(storage=self._storage, config=self._config)
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
            has_on_request=self._rtc is not None and self._rtc.on_request is not None,
        )

    def update_options(self, **options: Any) -> None:
        """Change options of ``AgentServer()`` before the server starts."""
        self._config.update(_core_options(options))

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


class _ServerHost:
    """What the core's server runtime opens on the host (``open(kind, args)``): the job
    processes and the user's callbacks. Which jobs to take, when they were taken and whether job
    processes start at all are the core's decisions."""

    def __init__(self, server: AgentServer) -> None:
        self._server = server
        self._proc_pool: ipc.ProcPool | None = None
        self._inference_executor: ipc.InferenceProcess | None = None

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
        """The job processes, when the core wants them."""
        os.environ.update(opts["env"])
        # must run before job processes spawn so they inherit the SSL_CERT_FILE fallback
        http_context._set_default_cert_env()
        logger.info(
            "starting worker", extra={"version": __version__, "rtc-version": rtc.__version__}
        )
        for p in Plugin.registered_plugins:
            logger.info("plugin registered", extra={"plugin": p.title, "version": p.version})
        door = self._server._rtc
        if not opts["job_processes"] or door is None:
            return
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
            self._inference_executor = ipc.InferenceProcess(
                runners=_InferenceRunner.registered_runners,
                mp_ctx=mp_ctx,
                loop=loop,
                http_proxy=opts["pool"]["http_proxy"],
            )
            logger.info("starting inference executor")
            await self._inference_executor.start()
            await self._inference_executor.initialize()
        storage = self._server._storage
        self._proc_pool = ipc.ProcPool(
            job_entrypoint_fnc=(
                CallThreadEntrypoint(door.entrypoint, storage) if storage else door.entrypoint
            ),
            session_end_fnc=door.on_session_end,
            simulation_end_fnc=door.on_simulation_end,
            loop=loop,
            job_executor_type=JobExecutorType(opts["job_executor_type"]),
            inference_executor=self._inference_executor,
            mp_ctx=mp_ctx,
            **opts["pool"],
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

    async def _metrics(self, _: Any) -> tuple[str, str]:
        return await asyncio.get_running_loop().run_in_executor(None, telemetry.metrics.exposition)

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

    def _job_status(self, proc: ipc.JobExecutor) -> None:
        if proc.running_job is not None and self._server._runtime is not None:
            self._server._runtime.job_status(proc.running_job.job.id, proc.status.value)

    # --- the user's callbacks

    @utils.log_exceptions(logger=logger)
    async def _job_request(self, req: dict[str, Any]) -> dict[str, Any]:
        """The voice door's ``on_request``, which the core asks only when there is one; a
        request it leaves unanswered or raises in, the core rejects."""
        answer: dict[str, Any] = {"decision": "unanswered"}

        async def on_reject(terminate: bool) -> None:
            answer.update(decision="terminate" if terminate else "reject")

        async def on_accept(args: JobAcceptArguments) -> None:
            answer.update(
                decision="accept",
                name=args.name,
                identity=args.identity,
                metadata=args.metadata,
                attributes=list((args.attributes or {}).items()),
            )

        door = self._server._rtc
        assert door is not None and door.on_request is not None
        job = agent.Job.FromString(req["job"]["proto"])
        await door.on_request(JobRequest(job=job, on_reject=on_reject, on_accept=on_accept))
        return answer

    async def _launch_job(self, info: dict[str, Any]) -> str:
        """A job the core took, with the timeline it stamped."""
        job = agent.Job.FromString(info["job"]["proto"])
        await self.launch(
            RunningJobInfo(
                accept_arguments=JobAcceptArguments(**info["accept_arguments"]),
                job=job,
                url=info["url"],
                token=info["token"],
                worker_id=info["worker_id"],
                fake_job=info["fake_job"],
                received_at=info["received_at"],
                accepted_at=info["accepted_at"],
                assigned_at=info["assigned_at"],
            )
        )
        return job.id  # `job_status` is keyed on the LiveKit job id

    async def _terminate_job(self, args: dict[str, Any]) -> None:
        if self._proc_pool is not None and (proc := self._proc_pool.get_by_job_id(args["job_id"])):
            await proc.aclose()

    async def _load(self, _: Any) -> float:
        """The user's ``load_fnc``, which the core asks only when there is one."""
        load_fnc: Callable[..., float] | None = self._server.load_fnc
        assert load_fnc is not None
        args = (self._server,) if inspect.signature(load_fnc).parameters else ()
        return await asyncio.get_running_loop().run_in_executor(None, load_fnc, *args)
