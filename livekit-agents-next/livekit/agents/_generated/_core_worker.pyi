# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Self

from livekit.agents._core import Role
from livekit.agents.http import _AsgiServer
from livekit.agents.ipc import JobExecutor, ProcPool as HostProcPool
from livekit.agents.ipc.child import _JobProc
from livekit.agents.worker import _ServerCallbacks

from .worker_ipc import StartJobFields
from .worker_options import ServerOptions
from .worker_server import EndpointRoute

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

class ByteStream:
    async def read(self, max_bytes: int = 65536) -> bytes: ...
    async def write(self, data: bytes | bytearray | memoryview | list[int]) -> None: ...
    def finish(self) -> None:
        """Idempotent; a stream already reset raises."""
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...

class _core_HttpServerCapacity:
    """Shared admission and payload reservations for a host's HTTP application server."""
    def __new__(cls, max_handlers: int, max_bytes: int) -> Self: ...
    @property
    def handlers(self) -> int: ...
    @property
    def payload_bytes(self) -> int: ...
    def admit(self, bytes: int) -> bool:
        """Atomically acquire a handler slot and its initial read/head reservation."""
    def reserve(self, bytes: int) -> bool: ...
    def release(self, bytes: int) -> None: ...
    def complete(self, bytes: int) -> None:
        """Release the handler and its initial reservation after all connection cleanup settles."""

def cpu_count() -> float:
    """Logical CPUs, fractional under a cgroup quota. Reads, starts nothing."""

class JobSupervisor:
    def __new__(
        cls,
        host: JobExecutor,
        *,
        thread: bool,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
    ) -> Self:
        """`thread`: the runner is a thread of this process, else a process of the host's start
        method.
        """
    async def initialize(self, *, asyncio_debug: bool, http_proxy: str | None) -> float:
        """The seconds the process took to be ready."""
    @property
    def initialized(self) -> bool: ...
    def _core_launched_process(
        self,
        pid: int,
        sentinel: int,
        method: str,
        owned: Sequence[int],
    ) -> None:
        """The launch's receipt, reported on the launch worker right after the OS created the child."""
    def _core_thread_exited(self) -> None:
        """A job thread returned, reported from that thread."""
    async def run(self) -> bool:
        """Launch, supervise and wait for the runner's end; whether one ran (a process reaped, a
        thread returned). Cancelling the wait leaves the runner owned.
        """
    async def _core_wait_exit(self) -> bool: ...
    async def _core_join(self) -> bool: ...
    async def aclose(self) -> None: ...
    async def kill(self) -> None: ...
    async def start_job(self, fields: StartJobFields) -> None: ...
    @property
    def killed(self) -> bool: ...

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
    def __new__(
        cls,
        host: HostProcPool,
        *,
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
    ) -> Self: ...
    @property
    def processes(self) -> list[JobExecutor]: ...
    @property
    def jobs_waiting(self) -> int: ...
    @property
    def target_idle_processes(self) -> int: ...
    @target_idle_processes.setter
    def target_idle_processes(self, n: int) -> None: ...
    async def start(self) -> None: ...
    async def launch_job(self, job_id: str, info: object) -> None: ...
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
        recording: bool = False,
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
    @property
    def observability_url(self) -> str | None: ...
    @property
    def buffer_logs(self) -> bool: ...
    def shutdown(self, reason: str, user_initiated: bool) -> None: ...
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
    def init_recording(self, options: Mapping[str, bool]) -> bool | None:
        """The redaction to record with when the cloud upload is to be set up now, `None` when not."""
    @property
    def recording_initialized(self) -> bool: ...
    @property
    def redaction(self) -> bool: ...
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
    ) -> None:
        """Start the job processes and the health listener and, unless `unregistered`, register;
        return once closed. `endpoints` receives each stream for `routes`.
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
