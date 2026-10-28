# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from asyncio import Future
from collections.abc import Mapping, Sequence

from livekit.agents.cli import _DevServer

from .._core import StartJobFields
from .worker_cli import CliArgs
from .worker_options import ServerOptions

class EndpointStream:
    """An exchange after its preamble, with independent reads and writes."""
    def read(self, max_bytes: int = 65536) -> Future[bytes]: ...
    def write(self, data: bytes | bytearray | memoryview | list[int]) -> Future[None]: ...
    def finish(self) -> None: ...
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...

class EndpointServer:
    """The host's ASGI application bridge and its active WebSockets."""
    def __init__(self) -> None: ...
    def serve(self, stream: object, app: object, request: object) -> Future[None]: ...
    def close_websockets(
        self,
        code: int = 1001,
        reason: str = "worker draining",
    ) -> Future[None]: ...

class _core_AsgiReceive:
    """The native ASGI receive callback, bound to one endpoint exchange."""
    def __call__(self) -> Future[object]: ...

class _core_AsgiSend:
    """The native ASGI send callback, bound to one endpoint exchange."""
    def __call__(self, message: object) -> Future[None]: ...

def job_timeline(
    received_at: float,
    accepted_at: float,
    assigned_at: float,
    launched_at: float,
    entrypoint_started_at: float,
    server_started_at: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, float]]]:
    """A job's dispatch timeline, for its root span: one event per known stage instant (unix
    nanoseconds) and the seconds between adjacent stages as attributes (they sum to the dispatch
    latency). A zero instant is unknown (a simulation, the console, a resumed job) and skipped.
    `server_started_at` is the server's own start (`JobState.started_at`, unix nanoseconds, or
    milliseconds or seconds should that change).
    """

def cpu_count() -> float:
    """Logical CPUs, fractional under a cgroup quota. Reads, starts nothing."""

def endpoint_url_base(ws_url: str, agent_name: str, deployment: str) -> str | None:
    """`https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
    the URL and the agent are known; a worker registered without a deployment is `default`.
    """

def parse_cli(argv: Sequence[str]) -> CliArgs: ...

def entrypoint_module(path: str | None) -> tuple[str, str]: ...

def format_log_record(
    style: str,
    record: object,
    message: str,
    asctime: str,
    exception: str | None,
    stack: str | None,
    ensure_ascii: bool,
) -> str:
    """Format one host logging record using its boundary adapter."""

async def dev_channel(
    cli_addr: str,
    agent_name: str,
    url: str,
    api_key: str,
    api_secret: str,
    server: _DevServer,
) -> None:
    """The development launcher connection."""

def ipc_pair() -> tuple[int, int]:
    """Create the control endpoints for process bootstrap."""

class ExitSignals:
    """The process launcher's signal and final event-loop cleanup owner."""
    def __init__(self, escalate: bool = True) -> None: ...
    def signal(self, exit_served: bool) -> str: ...
    def served(self) -> None: ...
    def draining(self) -> None: ...
    def closing(self) -> None:
        """Arm the native deadline before the host cancels tasks or closes its executor."""
    def closed(self) -> None:
        """Disarm the deadline only after the host event loop has actually closed."""

class ProcSupervisor:
    def __init__(
        self,
        fd: int,
        host: object,
        *,
        pid: int,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
        killable: bool = True,
    ) -> None: ...
    def initialize(self, *, asyncio_debug: bool, http_proxy: str | None) -> Future[float]: ...
    @property
    def initialized(self) -> bool: ...
    def _core_launched_process(
        self,
        pid: int,
        sentinel: int,
        method: str,
        owned: Sequence[int],
    ) -> None: ...
    def run(self, joined: object, host: object) -> Future[bool]: ...
    def _core_wait_exit(self) -> Future[bool]: ...
    def _core_join(self) -> Future[bool]: ...
    def aclose(self) -> Future[None]: ...
    def kill(self) -> Future[None]: ...
    def start_job(self, fields: StartJobFields) -> Future[None]: ...
    @property
    def killed(self) -> bool: ...

class IpcClient:
    def __init__(self, fd: int, initialize: object, *, process: bool = False) -> None: ...
    @property
    def asyncio_debug(self) -> bool: ...
    def serve(self, host: object) -> Future[None]: ...
    def stop(self) -> None: ...
    def aclose(self) -> Future[None]: ...

class ProcPool:
    def __init__(
        self,
        host: object,
        *,
        num_idle_processes: int,
        initialize_timeout: float,
        close_timeout: float,
    ) -> None: ...
    @property
    def processes(self) -> list[object]: ...
    @property
    def jobs_waiting(self) -> int: ...
    @property
    def target_idle_processes(self) -> int: ...
    @target_idle_processes.setter
    def target_idle_processes(self, n: int) -> None: ...
    def start(self) -> Future[None]: ...
    def launch_job(self, info: object) -> Future[None]: ...
    def join_jobs(self) -> Future[None]: ...
    def aclose(self) -> Future[None]: ...

class JobRun:
    def __init__(
        self,
        *,
        fake_job: bool,
        session_end_timeout: float,
        redaction: bool,
        url: str = "",
        recording: bool = False,
        session_end: object | None = None,
        session_aclose_timeout: float = 60.0,
    ) -> None: ...
    def add_shutdown_callback(self, callback: object) -> None: ...
    @property
    def observability_url(self) -> str | None: ...
    @property
    def buffer_logs(self) -> bool: ...
    def shutdown(self, reason: str, user_initiated: bool) -> None: ...
    def connected(self) -> None: ...
    def connect(self, connect: object) -> Future[None]: ...
    def add_participant_entrypoint(self, entrypoint: object, kinds: Sequence[int]) -> None: ...
    def participant_joined(
        self,
        ctx: object,
        participant: object,
        identity: str,
        kind: int,
    ) -> None: ...
    def participant_left(self, attributes: Mapping[str, str]) -> None: ...
    def api_call(self, call: object, name: str, console_result: object) -> Future[object]: ...
    def close_tasks(self) -> Future[None]: ...
    def init_recording(self, options: Mapping[str, bool]) -> bool | None: ...
    def claim_session(self, session: object) -> bool: ...
    @property
    def primary_session(self) -> object | None: ...
    @property
    def recording_initialized(self) -> bool: ...
    @property
    def redaction(self) -> bool: ...
    def run(self, host: object) -> Future[None]: ...

class ServerConfig:
    def __init__(self, options: ServerOptions) -> None: ...
    def update(self, options: ServerOptions) -> None: ...
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
    def cloud(self) -> bool: ...
    @property
    def drain_timeout(self) -> float: ...
    @property
    def simulation(self) -> bool: ...
    @simulation.setter
    def simulation(self, value: bool) -> None: ...

class ServerRuntime:
    """Returned host references are futures owned by the bootstrap loop."""
    def __init__(self, config: object, host: object) -> None: ...
    @property
    def worker_id(self) -> str: ...
    @property
    def draining(self) -> bool: ...
    @property
    def http_port(self) -> int: ...
    def run(
        self,
        *,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        publisher: bool,
        routes: object,
        has_load_fnc: bool,
        has_on_request: bool,
    ) -> Future[None]: ...
    def drain(self, timeout: float | None = None) -> Future[None]: ...
    def aclose(self) -> Future[None]: ...
    def job_status(self, job_id: str, status: str) -> None: ...
    def simulate_job(
        self,
        room: str,
        *,
        fake_job: bool,
        agent_identity: str | None = None,
        room_info: bytes | bytearray | memoryview | list[int] | None = None,
        token: str | None = None,
    ) -> Future[None]: ...
