"""Type stub for ``livekit.agents._core``, the pyo3 extension built from ``rust/crates/lk-agents-py``.

The generated stubs are star-imported below, written by ``cargo run -p lk-agents-codegen --
--write`` (from ``rust/``): ``_generated/_core_classes.pyi`` for the bindings expanded in place,
``_generated/_core_<subject>.pyi`` for the declared classes and functions. This file is
hand-written and carries the rest: the classes of
``lk-agents-py/src/{runtime,storage,turn}.rs``, whose requests and results are the pydantic models
of ``_generated/storage*.py``. The full name list is ``rust/crates/lk-agents-py/src/core_names.json``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Protocol

from ._generated._core_a2a import (
    _core_Credentials as _core_Credentials,
    _core_auth_grant_allows as _core_auth_grant_allows,
)
from ._generated._core_agent import *  # noqa: F401, F403
from ._generated._core_classes import *  # noqa: F401, F403
from ._generated._core_classes import ChatItem, StartJobFields
from ._generated._core_inference import *  # noqa: F401, F403
from ._generated._core_llm import *  # noqa: F401, F403
from ._generated._core_telemetry import *  # noqa: F401, F403
from ._generated._core_telemetry import (
    _core_SessionReportUpload as _core_SessionReportUpload,
    _core_TelemetryWorker as _core_TelemetryWorker,
    _core_loop_monitor_thresholds_from_env as _core_loop_monitor_thresholds_from_env,
)
from ._generated._core_storage import *  # noqa: F401, F403
from ._generated._core_worker import *  # noqa: F401, F403
from ._generated.storage import EventRow, InboxRow, StreamInfo
from ._generated.storage_run import ItemRow, LaneRow, TaskPage, TaskRow
from ._generated.storage_turn import Policy, TurnOptions, TurnResult
from ._generated.worker_options import ServerOptions

# --- runtime.rs ----------------------------------------------------------------------------------

class Role(Enum):
    Worker = ...
    Job = ...

def init(role: Role) -> None: ...
def cpu_count() -> float:
    """Logical CPUs, fractional under a cgroup quota; the worker sizes its defaults by it."""

# --- the worker ----------------------------------------------------------------------------------

class ServerHostProtocol(Protocol):
    """What ``ServerRuntime`` calls back into, in the generator's ``open(kind, args)`` shape: the
    worker's callbacks (``job_request``, ``launch_job``, ``terminate_job``, ``load``, ``endpoint``)
    and the server's (``start``: the job processes from the resolved options; ``started``,
    ``registered``, ``idle_processes``, ``worker_load``, ``health``, ``metrics``,
    ``close_websockets``,
    ``join_jobs``, ``close``). The awaitable runs on the loop that constructed the runtime and is
    cancelled when Rust drops the operation.
    """

    def open(self, kind: str, args: Any) -> Awaitable[Any]: ...

class EndpointStream:
    """One exchange stream of the endpoint data plane, positioned after its preamble."""

    def read(self, max_bytes: int = 65536) -> Awaitable[bytes]: ...
    def write(self, data: bytes) -> Awaitable[None]: ...
    def finish(self) -> None: ...
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...

class EndpointServer:
    """``AgentHttp``'s server half: one ASGI call per endpoint stream (an HTTP/1.1 exchange, or a
    WebSocket after an upgrade), and the open WebSockets to drain."""

    def __init__(self) -> None: ...
    def serve(self, stream: Any, app: Any, request: Any) -> Awaitable[None]: ...
    def close_websockets(
        self, code: int = 1001, reason: str = "worker draining"
    ) -> Awaitable[None]: ...

def endpoint_url_base(ws_url: str, agent_name: str, deployment: str) -> str | None: ...

class ServerConfig:
    """``AgentServer``'s options, resolved against the environment; writable until it starts."""

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
    """``AgentServer``'s lifecycle over a ``ServerHostProtocol``."""

    def __init__(self, config: ServerConfig, host: ServerHostProtocol) -> None: ...
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
        routes: list[dict[str, Any]],
        has_load_fnc: bool,
        has_on_request: bool,
    ) -> Awaitable[None]: ...
    def drain(self, timeout: float | None = None) -> Awaitable[None]: ...
    def aclose(self) -> Awaitable[None]: ...
    def job_status(self, job_id: str, status: str) -> None: ...
    def simulate_job(
        self,
        room: str,
        *,
        fake_job: bool,
        agent_identity: str | None = None,
        room_info: bytes | None = None,
        token: str | None = None,
    ) -> Awaitable[None]: ...

# --- the job process IPC (worker ipc/) -----------------------------------------------------------

def ipc_pair() -> tuple[int, int]:
    """One process's control socket: the parent's end and the child's, as descriptors the caller
    owns (the parent's for ``ProcSupervisor``, the child's for the process's ``IpcClient``)."""

class ProcHostProtocol(Protocol):
    """Bootstrap completion and queued diagnostics; native ownership lasts through reaping."""

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None: ...
    async def _core_launched(self) -> None: ...
    async def _core_launch_failed(self, cause: str) -> None: ...
    async def _core_completed(self, exitcode: int, pid: int) -> None: ...

class ProcSupervisor:
    """The worker's side of one job process, over the socket ``fd`` it owns."""

    def __init__(
        self,
        fd: int,
        host: ProcHostProtocol,
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
    def initialize(self, *, asyncio_debug: bool, http_proxy: str | None) -> Awaitable[float]:
        """Sends the ``Hello``; the seconds until the process was ready, ``asyncio.TimeoutError``
        when late (a process is then killed)."""
    @property
    def initialized(self) -> bool: ...
    def _core_launched_process(
        self, pid: int, sentinel: int, method: str, owned: list[int]
    ) -> None: ...
    def run(self, joined: Awaitable[None], host: ProcHostProtocol) -> asyncio.Future[bool]:
        """Owns the bootstrap operation and native process through OS exit and reaping."""
    def _core_join(self) -> Awaitable[bool]: ...
    def _core_wait_exit(self) -> Awaitable[bool]: ...
    def aclose(self) -> Awaitable[None]: ...
    def kill(self) -> Awaitable[None]: ...
    def start_job(self, fields: StartJobFields) -> Awaitable[None]:
        """Returns after the child accepts the typed job record."""
    @property
    def killed(self) -> bool: ...

class IpcClient:
    """The child's side of the parent's socket ``fd`` it owns: the initialize handshake (blocking,
    ``initialize(http_proxy)``), then the parent's messages served to a host: it starts one job
    and acks the shutdown."""

    def __init__(
        self, fd: int, initialize: Callable[[str | None], None], *, process: bool = False
    ) -> None: ...
    @property
    def asyncio_debug(self) -> bool: ...
    def serve(self, host: Any) -> Awaitable[None]: ...
    def stop(self) -> None: ...
    def aclose(self) -> Awaitable[None]: ...

class JobRun:
    """A job's run, for ``JobContext`` and ``ipc/job_proc_lazy_main.py``'s ``_JobProc``: what
    ends it, the entrypoint's supervision, the order and bounds of the shutdown (whose steps the
    host given to ``run`` runs, ``_core_*``), the participant entrypoints, the room connected
    once, the API calls in flight and the recording's setup."""

    def __init__(
        self,
        *,
        fake_job: bool,
        session_end_timeout: float,
        redaction: bool,
        url: str = "",
        recording: bool = False,
        session_end: Callable[..., Any] | None = None,
        session_aclose_timeout: float = 60.0,
    ) -> None: ...
    def add_shutdown_callback(self, callback: Callable[..., Awaitable[None]]) -> None:
        """``callback(reason)``, or ``callback()`` when it takes no argument, runs as the job
        shuts down."""
    @property
    def observability_url(self) -> str | None: ...
    @property
    def buffer_logs(self) -> bool:
        """The job buffers its logs from the start, for the recording to upload them."""
    def shutdown(self, reason: str, user_initiated: bool) -> None: ...
    def connected(self) -> None: ...
    def connect(self, connect: Callable[[], Awaitable[None]]) -> Awaitable[None]: ...
    def add_participant_entrypoint(self, entrypoint: Any, kinds: list[int]) -> None: ...
    def participant_joined(self, ctx: Any, participant: Any, identity: str, kind: int) -> None: ...
    def participant_left(self, attributes: dict[str, str]) -> None: ...
    def api_call(
        self, call: Callable[[], Awaitable[Any]], name: str, console_result: Any
    ) -> asyncio.Future[Any]: ...
    def close_tasks(self) -> Awaitable[None]: ...
    def init_recording(self, options: dict[str, bool]) -> bool | None: ...
    def claim_session(self, session: Any) -> bool: ...
    @property
    def primary_session(self) -> Any: ...
    @property
    def recording_initialized(self) -> bool: ...
    @property
    def redaction(self) -> bool: ...
    def run(self, host: Any) -> Awaitable[None]: ...

def job_timeline(
    received_at: float,
    accepted_at: float,
    assigned_at: float,
    launched_at: float,
    entrypoint_started_at: float,
    server_started_at: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, float]]]:
    """A job's dispatch timeline for its root span: the known stage instants as events (unix
    nanoseconds), the seconds between adjacent stages as attributes."""

class ProcPool:
    """The warm job processes of ``ipc/proc_pool.py``'s ``ProcPool`` (``host``), which makes the
    executors (``_core_create``), reports a failed initialization (``_core_init_failed``) and
    emits the pool's events."""

    def __init__(
        self, host: Any, *, num_idle_processes: int, initialize_timeout: float, close_timeout: float
    ) -> None: ...
    @property
    def processes(self) -> list[Any]: ...
    @property
    def jobs_waiting(self) -> int: ...
    @property
    def target_idle_processes(self) -> int: ...
    @target_idle_processes.setter
    def target_idle_processes(self, n: int) -> None: ...
    def start(self) -> Awaitable[None]: ...
    def launch_job(self, info: Any) -> Awaitable[None]: ...
    def join_jobs(self) -> Awaitable[None]:
        """Waits until every process a job was launched on exited."""
    def aclose(self) -> Awaitable[None]: ...

# --- storage.rs ----------------------------------------------------------------------------------
#
# Every request is the model of its name, or a dict in the model's shape (its ``model_dump``);
# every result, rows included, is an instance of its model. An ``Awaitable`` runs on the loop
# that constructed the object and is cancelled when Rust drops the operation.

class Streams:
    """The response streams of one storage (``Storage.streams()``)."""

    def create(self, ttl_seconds: int | None = None) -> Awaitable[StreamInfo]: ...
    def append(self, stream_id: str, record: bytes) -> None:
        """Queued: kept in order; a failure is logged and does not fail the turn."""

    def close(self, stream_id: str) -> Awaitable[None]:
        """Once the stream's queued frames are kept."""

    def delete(self, stream_id: str) -> Awaitable[None]: ...
    def read(self, stream_id: str, after: str | None = None, live: bool = False) -> StreamReader:
        """From ``after`` (an offset; ``None`` is the beginning); ``live`` follows the tail until
        the stream closes."""

class StreamReader:
    """A read position over one stream."""

    def next(self) -> Awaitable[bytes | None]:
        """The next frame, or ``None``: the tail of a catch-up read, the close of a live one."""

    def next_offset(self) -> Awaitable[str]: ...

# --- turn.rs -------------------------------------------------------------------------------------

class Thread:
    """One open thread (plan 5.2), from ``Storage.open``: leases and waiter marks carry
    ``holder`` plus a handle id of their own."""

    @property
    def id(self) -> str: ...
    @property
    def holder(self) -> str: ...
    def lanes(self) -> Awaitable[list[LaneRow]]: ...
    def events(self, after_id: int = 0, limit: int = 100) -> Awaitable[list[EventRow]]: ...
    def history(self, lane: str) -> Awaitable[list[ItemRow]]:
        """A lane's history as stored, in order."""

    def tasks(self, lane: str, limit: int = 50) -> Awaitable[list[TaskRow]]:
        """A lane's task rows as records, newest first."""

    def task(self, task_id: str) -> Awaitable[TaskRow | None]:
        """One task, whichever lane it is on; ``None`` when the thread has none of that id."""

    def list_tasks(
        self,
        status: str | None = None,
        updated_after: int = 0,
        page_size: int = 50,
        page_token: str | None = None,
    ) -> Awaitable[TaskPage]:
        """One page of every lane's tasks, last updated first (A2A ``tasks/list``)."""

    def lane_info(self, lane: str) -> Awaitable[LaneRow | None]:
        """A lane's row, ``None`` before its first run."""

    def inbox(self, lane: str, pending: bool = False) -> Awaitable[list[InboxRow]]:
        """A lane's inbox rows in arrival order: every one, or the pending ones."""

    def cancel_task(
        self, lane: str, task_id: str, metadata: dict[str, Any]
    ) -> Awaitable[TaskRow | None]:
        """``tasks/cancel`` (plan 5.5): the row ``canceled`` with ``metadata`` merged in;
        ``None`` when the task is not the lane's."""

    def open_run(
        self,
        lane: str,
        run_id: str,
        *,
        channel: str | None = None,
        wait_timeout: float | None = None,
    ) -> Awaitable[Run]:
        """Takes the lane for one run (plan 5.2), waiting up to ``wait_timeout`` seconds for
        another holder's lease."""

    def close(self) -> None: ...

class Run:
    """One run the host drives itself (the voice lane): what it loaded, then one ``commit`` of
    the history and state it ends with, or ``abandon``."""

    @property
    def version(self) -> int:
        """The version loaded: the commit's fence."""

    @property
    def closed(self) -> bool:
        """The run's id already committed: ``commit`` only releases."""

    @property
    def history(self) -> list[ItemRow]: ...
    @property
    def state(self) -> Any: ...
    def commit(self, history: list[ItemRow], state: Any = None) -> Awaitable[int]:
        """Writes what changed since the load (``None`` state keeps the stored one), then
        releases; the version after."""

    def abandon(self, outcome: str, reason: str) -> Awaitable[None]:
        """``run_ended`` with ``outcome`` and ``reason``, and the release."""

class TurnHostProtocol(Protocol):
    """What ``Turn`` calls back into: ``_core_handle`` receives a ``storage_turn.TurnContext``
    dict and returns the turn's event stream (``storage.context.TurnStream``)."""

    def _core_handle(self, ctx: dict[str, Any]) -> Any: ...

class Turn:
    """One text door's turns (``TurnRunner``); ``options`` left out or ``None`` is the Rust
    defaults, and so is every field left out of it."""

    def __init__(
        self,
        thread: Thread,
        lane: str,
        host: TurnHostProtocol,
        options: TurnOptions | None = None,
    ) -> None: ...
    @property
    def lane(self) -> str: ...
    def policy_for(self, channel: str) -> Policy: ...
    def run(
        self,
        message: dict[str, Any],
        task_id: str | None = None,
    ) -> Awaitable[TurnResult]:
        """One incoming A2A message through its policy. Raises ``RunRejected`` under
        ``reject``, ``asyncio.CancelledError`` when the handler was cancelled, the storage
        errors otherwise."""

# --- the voice nodes (lk-agents-py nodes.rs) -----------------------------------------------------

class ItemStream:
    def __aiter__(self) -> ItemStream: ...
    def __anext__(self) -> Awaitable[Any]: ...
    def aclose(self) -> Awaitable[None]: ...

class AgentNodes:
    def __init__(self, context: Any, host: Any) -> None: ...

def tts_node(tts: Any, text: Any, *, conn_options: Any) -> ItemStream: ...
def stt_node(stt: Any, audio: Any, *, conn_options: Any) -> ItemStream: ...

# --- lk-agents-abi llm/to_dict.rs ----------------------------------------------------------------

def chat_ctx_to_dict(
    items: list[ChatItem],
    *,
    exclude_image: bool = True,
    exclude_audio: bool = True,
    exclude_timestamp: bool = True,
    exclude_function_call: bool = False,
    exclude_metrics: bool = False,
    exclude_config_update: bool = False,
    serialize_media: Callable[[Any], Any] | None = None,
    strip_expr_markup: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """``ChatContext.to_dict``: ``{"items": [...]}``, each item's JSON dump. ``serialize_media``
    renders a frame (required only when media is kept); ``strip_expr_markup`` strips assistant
    messages' markup."""
