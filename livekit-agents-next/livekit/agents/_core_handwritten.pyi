"""Stubs of the ``livekit.agents._core`` names whose pyo3 code is still handwritten.

``_core.pyi`` (generated) re-exports exactly the names this file defines that
``lk-agents-codegen``'s ``surface::HANDWRITTEN`` lists; ``core_names.json`` checks them against the
extension's registrations. A family whose pyo3 code moves to ABI declarations leaves both, and its
stub is then generated. The ``*Protocol`` classes describe host objects and are not ``_core`` names.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Protocol

from ._generated._core_classes import (
    ChatItem,
    StartJobFields,
    Tagger,
    UploadGate,
)
from ._generated.storage_class import (
    EventRow,
    InboxRow,
    ItemRow,
    LaneRow,
    StreamInfo,
    TaskPage,
    TaskRecord,
)
from ._generated.storage_context import ContextScope, ContextScopeWrite
from ._generated.storage_options import Policy, TurnOptions
from ._generated.storage_turn import TurnContext, TurnResult
from ._generated.telemetry import SpanUpdateWire
from ._generated.worker_options import ServerOptions

# --- runtime.rs ----------------------------------------------------------------------------------

class Role(Enum):
    Worker = ...
    Job = ...

def init(role: Role) -> None: ...

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

    def tasks(self, lane: str, limit: int = 50) -> Awaitable[list[TaskRecord]]:
        """A lane's task rows as records, newest first."""

    def task(self, task_id: str) -> Awaitable[TaskRecord | None]:
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
    ) -> Awaitable[TaskRecord | None]:
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
    @property
    def context_scopes(self) -> list[ContextScope]: ...
    def commit(
        self,
        history: list[ItemRow],
        state: Any = None,
        *,
        scope_writes: list[ContextScopeWrite] | None = None,
    ) -> Awaitable[int]:
        """Writes what changed since the load (``None`` state keeps the stored one), then
        releases; the version after."""

    def abandon(self, outcome: str, reason: str) -> Awaitable[None]:
        """``run_ended`` with ``outcome`` and ``reason``, and the release."""

class TurnHostProtocol(Protocol):
    """What ``Turn`` calls back into: ``_core_handle`` receives a ``storage_turn.TurnContext``
    dict and returns the turn's event stream (``storage.context.TurnStream``)."""

    async def _core_handle(self, ctx: TurnContext) -> Any: ...

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

def resolve_host_ref(id: int) -> Any:
    """The object the host interned under ``id``; read, not released."""

def run_shutdown_callbacks(
    callbacks: list[Callable[..., Awaitable[None]]], reason: str, per_callback: float = 10.0
) -> Awaitable[None]: ...

# --- metrics and telemetry -----------------------------------------------------------------------

def session_records(
    gate: UploadGate,
    report: dict[str, Any],
    tagger: Tagger,
    project_item: Callable[[Any], Any],
    capture_content: bool = False,
) -> list[Any]:
    """Upload records for a completed session; ``project_item`` serializes chat items."""

# --- tokenize ------------------------------------------------------------------------------------

# --- inference -----------------------------------------------------------------------------------

def stt_keyterms_extra_for_model(
    model: str | None = None,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    session_keyterms: list[str] | None = None,
) -> dict[str, Any] | None: ...
def stt_capabilities(
    model: str | None = None,
    *,
    fallback: list[str] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any: ...

class EotThresholdOptions:
    """Shared between a detector and its stream: a stream's cloud-to-local fallback shows here."""

    def __init__(
        self,
        model: str,
        overrides: dict[str, float] | None = None,
        backchannel_overrides: dict[str, float] | None = None,
    ) -> None: ...
    @property
    def model(self) -> str: ...
    @property
    def overrides(self) -> dict[str, float] | None: ...
    @property
    def backchannel_overrides(self) -> dict[str, float] | None: ...
    @property
    def thresholds(self) -> dict[str, float]: ...
    @property
    def default_threshold(self) -> float | None: ...
    def lookup(self, language: str | None = None) -> float | None: ...
    def lookup_backchannel(self, language: str | None = None) -> float | None: ...
    def supports(self, language: str | None = None) -> bool: ...
    def update_overrides(self, overrides: dict[str, float] | None = None) -> None: ...
    def update_backchannel_overrides(self, overrides: dict[str, float] | None = None) -> None: ...
    def update_defaults(
        self,
        server_thresholds: dict[str, float],
        server_default: float,
        backchannel_thresholds: dict[str, float] | None = None,
        backchannel_default: float = 0.0,
    ) -> None: ...

class EotStream:
    def __init__(
        self,
        model: str,
        thresholds: EotThresholdOptions,
        *,
        local_fallback: bool,
        base_url: str,
        api_key: str,
        api_secret: str,
        headers: Callable[[], Any],
        sample_rate: int,
        conn_options: Any,
        on_metrics: Callable[[dict[str, Any]], None] | None = None,
    ) -> None: ...
    @property
    def model(self) -> str: ...
    @property
    def streaming(self) -> bool: ...
    @property
    def is_fallback(self) -> bool: ...
    @property
    def is_degraded(self) -> bool: ...
    def predict(self) -> Awaitable[Any]: ...
    def cancel_inference(self, *, timed_out: bool = False) -> None: ...
    def flush(self, reason: str | None = None) -> None: ...
    def push_audio(self, frame: Any) -> None: ...
    def end_input(self) -> None: ...

class InterruptionDetector:
    def __init__(
        self,
        *,
        threshold: float | None = None,
        min_interruption_duration: float = ...,
        max_audio_duration: float = ...,
        audio_prefix_duration: float = ...,
        detection_interval: float = ...,
        inference_timeout: float = ...,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        headers: Any = None,
        conn_options: Any = ...,
        token_ttl_secs: int = 600,
        http_session: Any = None,
    ) -> None: ...
    def update_options(
        self, *, threshold: float | None = None, min_interruption_duration: float | None = None
    ) -> None: ...
    def recv_event(self) -> Awaitable[dict[str, Any] | None]: ...
    def stream(self) -> Awaitable[InterruptionDetectorStream]: ...

class InterruptionDetectorStream:
    def push_audio(self, frame: Any) -> None:
        """One audio frame; synchronous (50 fps)."""

    def push_event(
        self,
        kind: str,
        *,
        started_at: float = 0.0,
        ended_at: float = 0.0,
        speech_duration: float = 0.0,
        agent_ended: bool = False,
        user_speaking_span: Any = None,
    ) -> None: ...
    def flush(self) -> None: ...
    def end_input(self) -> None: ...
    def recv(self) -> Awaitable[dict[str, Any] | SpanUpdateWire | None]:
        """The next event (``overlapping_speech``, ``error`` or ``span_update``), ``None`` at the end."""

    def aclose(self) -> Awaitable[None]: ...

def interruption_metrics(event: Any, wall: float, model: str, provider: str) -> dict[str, Any]: ...

class FixedGate:
    def __init__(
        self,
        silence: float,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
    ) -> None: ...
    def update(self, frame: Any) -> bool: ...
    def deactivate(self) -> None: ...

class AdaptiveNoiseGate:
    def __init__(
        self,
        *,
        activation_ratio: float = 3.0,
        deactivation_ratio: float = 1.8,
        min_silence_duration: float = 0.5,
        window: float = 10.0,
    ) -> None: ...
    def update(self, frame: Any) -> bool: ...
    def deactivate(self) -> None: ...

class DuplexRealtimeModel:
    def __init__(self, host: Any, *, audio_timeout: float = 0.8) -> None: ...
    def capabilities(self) -> Any:
        """The realtime capabilities the model runs under."""
