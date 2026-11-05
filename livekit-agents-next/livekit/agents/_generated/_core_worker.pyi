# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Self

from livekit.agents._core import Role
from livekit.agents.http import _AsgiServer
from livekit.agents.ipc import JobLauncher, ProcPool as HostProcPool
from livekit.agents.ipc.child import _JobProc
from livekit.agents.worker import _ServerCallbacks

from .._core import _core_A2ADoor
from .worker_endpoint import ExchangeEvent, ExchangeMessage, RequestHead
from .worker_ipc import JobRunnerOptions, JobStatus, StartJobFields
from .worker_options import ServerOptions
from .worker_server import EndpointRoute, RunningJobInfo

def room_join_token(
    api_key: str,
    api_secret: str,
    identity: str,
    room: str,
    *,
    agent: bool = False,
    ttl: float | None = None,
) -> str:
    """A token joining `room` as `identity`, publishing, subscribing and updating its own
    metadata; `agent` joins as an agent. `ttl` in seconds; the signer's default when `None`.
    """

class HttpExchange:
    """One request and its response, or one WebSocket connection. The worker owns parsing, framing,
    limits, deadlines and closing; the binding reads events and sends messages.
    """
    @property
    def head(self) -> RequestHead: ...
    async def receive(self) -> ExchangeEvent:
        """The next request event. After the request body ends it waits for the response to
        complete, then reports `Disconnected`.
        """
    async def send(self, message: ExchangeMessage) -> None: ...

def cpu_count() -> float:
    """Logical CPUs, fractional under a cgroup quota. Reads, starts nothing."""

class JobLaunch:
    """One launch of a job runner, as the launcher reports it."""
    @property
    def thread(self) -> bool:
        """The runner is a thread of this process, else a process of the launcher's start method."""
    def process_started(self, pid: int, sentinel: int, method: str, owned: Sequence[int]) -> None:
        """The process's receipt, reported on the launch worker right after the OS created it."""
    def thread_exited(self) -> None:
        """The job thread returned, reported from that thread."""

class JobSupervisor:
    """One job runner, a process or (`options.thread`) a thread of this process, and its one job:
    the worker launches it through `launcher`, initializes it, hands it the job, pings it,
    samples its memory, closes it and waits for its end, whether or not the host still awaits.

    `status` is `None` until a job was launched, then `RUNNING` until the runner ended, then
    `SUCCESS` (exit code 0 and a clean control channel) or `FAILED`. `running_job` is set once
    the runner took its job and is kept after the runner ended. `aclose()` asks an initialized
    runner to shut down and stops it when late (`close_timeout`); `kill()` stops it at once. A
    thread cannot be stopped: both wait for it to return, and it has no pong deadline nor memory
    limit.
    """
    def __new__(cls, launcher: JobLauncher, options: JobRunnerOptions) -> Self: ...
    @property
    def id(self) -> str:
        """`PCEXEC_...` for a process, `THEXEC_...` for a thread."""
    @property
    def thread(self) -> bool: ...
    @property
    def started(self) -> bool:
        """`start()` was called."""
    @property
    def initialized(self) -> bool:
        """The runner answered its initialization."""
    @property
    def closing(self) -> bool:
        """`aclose()` or `kill()` was called: it starts no runner and takes no job."""
    @property
    def killed(self) -> bool:
        """The runner was stopped by force (a process killed)."""
    @property
    def pid(self) -> int | None:
        """The process's id, once the OS created it; `None` for a thread, and for a forkserver child
        whose id the forkserver has not reported.
        """
    @property
    def exitcode(self) -> int | None:
        """The runner's exit code once its supervision is over (negative: the signal that ended a
        Unix process).
        """
    @property
    def status(self) -> JobStatus | None: ...
    @property
    def running_job(self) -> RunningJobInfo | None: ...
    @property
    def initialize_time(self) -> float | None:
        """The seconds the runner took to be ready, once it was."""
    async def start(self) -> None:
        """Launch the runner; returns once it exists. Raises once closing, when already started, or
        with the launcher's own error when nothing was launched. A launch failing after the OS
        created the process is logged with its cause and the process is stopped.
        """
    async def initialize(self) -> float:
        """Send the runner its `Hello` and wait for it to be ready: the seconds it took. A late runner
        is stopped (`TimeoutError`).
        """
    async def launch_job(self, info: RunningJobInfo) -> None:
        """Hand the ready runner its one job, stamped `launched_at` now; the job continues the
        caller's current span. Raises when it is not ready or already has a job.
        """
    async def aclose(self) -> None:
        """Close it gracefully (see the class) and wait for its cleanup; at once when not started."""
    async def kill(self) -> None:
        """Stop it at once and wait for its cleanup; at once when not started."""
    async def join(self) -> bool:
        """Wait for the end of its supervision, its completion reported to the launcher: whether a
        runner ran. `False` at once when it was never started.
        """

class IpcClient:
    def __new__(
        cls,
        fd: int,
        initialize: Callable[[str | None], None],
        *,
        process: bool = False,
    ) -> Self:
        """The handshake on the parent's socket, blocking before the process's loop exists."""
    @property
    def asyncio_debug(self) -> bool: ...
    async def serve(self, host: _JobProc) -> None:
        """Serve the parent's messages until it goes, its shutdown ends the process, or `stop()`."""
    def stop(self) -> None: ...
    async def aclose(self) -> None: ...

class ProcPool:
    """The warm job processes: the pool keeps `num_idle_processes` ready, hands each job the next one
    and retries a failed spawn or launch. Each process is a `JobSupervisor` with `options`.
    """
    def __new__(
        cls,
        host: HostProcPool,
        *,
        num_idle_processes: int,
        options: JobRunnerOptions,
    ) -> Self: ...
    @property
    def processes(self) -> list[JobSupervisor]:
        """The processes the pool owns, from their creation until their cleanup."""
    @property
    def active_jobs(self) -> list[RunningJobInfo]:
        """The jobs whose process runs them now (`status` `RUNNING`)."""
    def get_by_job_id(self, job_id: str) -> JobSupervisor | None:
        """The pool's process that took the job `job_id`."""
    @property
    def jobs_waiting(self) -> int: ...
    @property
    def target_idle_processes(self) -> int: ...
    @target_idle_processes.setter
    def target_idle_processes(self, n: int) -> None: ...
    async def start(self) -> None: ...
    async def launch_job(self, info: RunningJobInfo) -> None:
        """The job continues the caller's current span."""
    async def join_jobs(self) -> None:
        """Waits until every process a job was launched on exited (the drain)."""
    async def aclose(self) -> None: ...

class JobRun:
    """A job's run: what ends the job (the first reason wins), the entrypoint's supervision, the
    order and bounds of the shutdown, the participant entrypoints, the room connected once, the
    application calls in flight and the recording's setup.
    """
    def __new__(
        cls,
        *,
        fake_job: bool,
        session_end_timeout: float,
        redaction: bool,
        url: str = "",
        session_end: str | None = None,
        session_aclose_timeout: float = 60.0,
        observability_url: str | None = None,
        api_key: str = "",
        api_secret: str = "",
    ) -> Self:
        """`session_end` names the door's `on_session_end`, when it has one. `observability_url`
        overrides where the job's telemetry and recording go (`LIVEKIT_OBSERVABILITY_URL`);
        `api_key` and `api_secret` sign its Cloud telemetry.
        """
    def add_shutdown_callback(
        self,
        callback: object,
        takes_reason: bool,
        framework: bool,
        name: str,
    ) -> None:
        """`callback(reason)` when it takes the reason, else `callback()`, as the job shuts down; a
        framework callback runs in the shutdown's span, any other in its own span named `name`.
        """
    def start_trace(self, fields: StartJobFields) -> None:
        """Opens the job's root span for the job the parent handed over, before its run starts."""
    def shutdown(self, reason: str) -> None:
        """The application ends the job."""
    def room_disconnected(self) -> None: ...
    def connected(self) -> None: ...
    async def connect(self, connect: Callable[[], Awaitable[object]]) -> None:
        """Awaits `connect()` once: a call while it runs waits for it, a call after it connected
        returns at once, a failed one leaves the next call to try again.
        """
    def add_participant_entrypoint(
        self,
        entrypoint: Callable[[object, object], Awaitable[None]],
        kinds: Sequence[int],
        name: str,
    ) -> None:
        """`entrypoint` (named `name`) runs for each joining participant of one of `kinds`."""
    def participant_joined(
        self,
        job: object,
        participant: object,
        identity: str,
        kind: int,
    ) -> None:
        """A participant joined, or was in the room when it connected; `job` is the `JobContext`."""
    def participant_left(self, attributes: Mapping[str, str]) -> None: ...
    async def api_call(
        self,
        call: Callable[[], Awaitable[object]],
        name: str,
        console_result: object,
    ) -> object:
        """An application API call (`delete_room`, ...): its failure is logged whether or not the
        caller awaits it, and it is cancelled once the job ends. A console job makes no call: it
        resolves to `console_result`.
        """
    async def close_tasks(self) -> None:
        """Cancels the API calls still running and waits for them."""
    def init_recording(self, options: Mapping[str, bool]) -> None:
        """The session's recording options: the job registers with the exporter's Cloud destination
        once, at the first call.
        """
    async def run(self, host: _JobProc) -> None:
        """The entrypoint, then the shutdown sequence once the job ends; `host` runs each step in the
        context this call captured.
        """

def ipc_pair() -> tuple[int, int]: ...

def init(role: Role) -> None: ...

class ServerConfig:
    def __new__(cls, options: ServerOptions) -> Self:
        """`options` left empty read the environment (`http_proxy`: `HTTPS_PROXY`, else
        `HTTP_PROXY`).
        """
    def update(self, options: ServerOptions) -> None:
        """Before the server starts only."""
    @property
    def ws_url(self) -> str: ...
    @property
    def api_key(self) -> str: ...
    @property
    def api_secret(self) -> str: ...
    @property
    def agent_name(self) -> str: ...
    @property
    def deployment(self) -> str: ...
    @property
    def cloud(self) -> bool:
        """Hosted on Cloud (`LIVEKIT_WORKER_TOKEN` set)."""
    @property
    def drain_timeout(self) -> float: ...
    @property
    def shutdown_timeout(self) -> float: ...
    @property
    def simulation(self) -> bool: ...
    @simulation.setter
    def simulation(self, value: bool) -> None: ...
    def endpoint_url(self, endpoint: str) -> str | None:
        """The address `endpoint` publishes in its card: its Cloud URL when Cloud routes it; its
        path when this worker serves it, put on the address the request reached.
        """
    def delegation_url(self, endpoint: str) -> str:
        """Where this worker delegates to its own `endpoint`: Cloud, in the worker's deployment, or
        the worker's public listener. Raises before the server knows which.
        """

class ServerRuntime:
    def __new__(cls, config: ServerConfig, host: _ServerCallbacks) -> Self: ...
    @property
    def worker_id(self) -> str: ...
    @property
    def draining(self) -> bool: ...
    @property
    def http_port(self) -> int: ...
    async def run(
        self,
        *,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        routes: Sequence[EndpointRoute],
        has_load_fnc: bool,
        endpoints: _AsgiServer,
        text_doors: Sequence[_core_A2ADoor],
    ) -> None:
        """Start the job processes and the health listener and, unless `unregistered`, register;
        return once closed. The worker serves each `text` route with its door from `text_doors`,
        whose handlers run in the context this call captured; `endpoints` receives every other
        stream for `routes`.
        """
    async def drain(self, timeout: float | None = None) -> None:
        """`None` (or `0`) waits without a deadline; past it raises `TimeoutError`."""
    async def aclose(self) -> None: ...
    def job_status(self, job_id: str, status: str) -> None:
        """`"running"`, `"failed"` or `"success"`."""
    async def simulate_job(
        self,
        room: str,
        *,
        fake_job: bool,
        agent_identity: str | None = None,
        room_info: bytes | bytearray | memoryview | list[int] | None = None,
        token: str | None = None,
    ) -> None:
        """A job in `room` with no dispatch: the console's, a notebook's."""
