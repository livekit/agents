"""Type stub for ``livekit.agents._core``, the pyo3 extension built from ``rust/crates/lk-agents-py``.

Two files make the stub. ``_generated/_core_classes.pyi`` (star-imported below) is written by
``cargo run -p lk-agents-codegen -- --write`` (from ``rust/``) for every ``_core`` class and
function the binding generates. This file is hand-written and carries the rest: the classes of
``lk-agents-py/src/{runtime,storage,turn}.rs``, whose requests and results are the pydantic models
of ``_generated/storage*.py``. The full name list is ``rust/crates/lk-agents-py/src/core_names.json``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from enum import Enum
from typing import Any, Protocol

from ._generated import storage_turn
from ._generated._core_classes import *  # noqa: F401, F403
from ._generated.storage import (
    AcquireRequest,
    ChatItemRow,
    CommitRequest,
    CommitResult,
    DatabaseInfo,
    DrainCheck,
    Duplicate,
    EventRow,
    Fence,
    InboxRow,
    ItemWrite,
    LaneRow,
    Loaded,
    Received,
    ReceiveRequest,
    RunEndedRequest,
    StreamInfo,
    SubmittedTask,
    TaskRow,
)
from ._generated.storage_run import Item, LaneInfo, TaskRecord
from ._generated.storage_threads import ThreadInfo, ThreadItemPage, ThreadPage, ThreadTask
from ._generated.storage_turn import HandlerOutput, Policy, TurnOptions, TurnResult

# --- runtime.rs ----------------------------------------------------------------------------------

class Role(Enum):
    Worker = ...
    Job = ...
    Inference = ...

def init(role: Role) -> None: ...
def cpu_count() -> float:
    """Logical CPUs, fractional under a cgroup quota; the worker sizes its defaults by it."""

# --- the worker ----------------------------------------------------------------------------------

class WorkerHostProtocol(Protocol):
    """What ``Worker`` calls back into, in the generator's ``open(kind, args)`` shape.

    ``kind`` is one of ``"job_request"`` (args: a ``JobRequest`` dict, result: a decision dict),
    ``"launch_job"`` (a ``RunningJobInfo`` dict, result: the LiveKit job id), ``"terminate_job"``
    (``{"job_id"}``), ``"load"`` (``{}``, result: a float or ``None`` for the CPU monitor) and
    ``"endpoint"`` (the preamble fields plus ``"stream"``, an ``EndpointStream``; served to
    completion). The awaitable runs on the loop that constructed the worker and is cancelled when
    Rust drops the operation.
    """

    def open(self, kind: str, args: dict[str, Any]) -> Awaitable[Any]: ...

class EndpointStream:
    """One exchange stream of the endpoint data plane, positioned after its preamble."""

    def read(self, max_bytes: int = 65536) -> Awaitable[bytes]: ...
    def write(self, data: bytes) -> Awaitable[None]: ...
    def finish(self) -> None: ...
    def reset(self, code: int) -> None: ...
    def stop(self, code: int) -> None: ...

class Worker:
    def __init__(self, options: dict[str, Any], host: WorkerHostProtocol) -> None: ...
    def run(self) -> Awaitable[None]: ...
    def drain(self, timeout: float | None = None) -> Awaitable[None]: ...
    def next_notification(self) -> Awaitable[dict[str, Any]]: ...
    def close(self) -> None: ...
    def job_status(self, job_id: str, status: int) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...

# --- storage.rs ----------------------------------------------------------------------------------
#
# Every request is the model of its name, or a dict in the model's shape (its ``model_dump``);
# every result, rows included, is an instance of its model. An ``Awaitable`` runs on the loop
# that constructed the object and is cancelled when Rust drops the operation.

class Storage:
    """The root: credentials and the agentstore URL (plan 13.4), the public API every request
    goes through, derived from ``LIVEKIT_URL``; ``api_key`` and ``api_secret`` default to ``LIVEKIT_API_KEY`` and
    ``LIVEKIT_API_SECRET``. Lease timing is in seconds, ``None`` for the Rust defaults."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: float | None = None,
        renew_interval: float | None = None,
    ) -> None: ...
    @property
    def url(self) -> str:
        """The public API agentstore is reached through."""

    def open(self, database_id: str) -> Awaitable[ThreadDb]:
        """One connection to the session's database, with its schema version read."""

    @property
    def streams(self) -> Streams: ...
    @property
    def threads(self) -> Threads: ...
    def create_database(self, ttl_seconds: int = 0) -> Awaitable[DatabaseInfo]:
        """``DatabaseService.CreateDatabase``, for development and tests."""

    def delete_database(self, database_id: str) -> Awaitable[None]: ...

class ThreadDb:
    """One session's database over one connection; the typed statements of plan 13.5. Every
    later call after ``close()`` raises ``StorageUnavailable``."""

    @property
    def database_id(self) -> str: ...
    @property
    def tip(self) -> int:
        """The durable commit sequence last observed on this connection."""

    def close(self) -> None: ...
    def acquire(self, request: AcquireRequest) -> Awaitable[Loaded | None]:
        """The acquire and the load, one batch (plan 3.3.2); ``None`` when another holder has
        the lease."""

    def release(self, lane: str, lease_token: str) -> Awaitable[bool]: ...
    def renew(self, lane: str, holder: str, lease_token: str) -> Awaitable[bool]:
        """One renewal; ``False`` when the lease is no longer this holder's."""

    def start_renewal(self, lane: str, holder: str, lease_token: str) -> Lease: ...
    def commit(self, request: CommitRequest) -> Awaitable[CommitResult]:
        """The fenced commit (plan 5.2); raises ``VersionConflict`` when the fence matched no
        row. A ``None`` fence is the ``concurrent`` policy's append-only commit."""

    def run_ended(self, request: RunEndedRequest) -> Awaitable[None]: ...
    def receive(self, request: ReceiveRequest) -> Awaitable[Received | Duplicate]:
        """The inbox receive (plan 3.3.4)."""

    def insert_submitted_task(self, task: SubmittedTask) -> Awaitable[bool]:
        """``True`` when inserted, ``False`` when the task or the message already had a row."""

    def drain_check(self, lane: str, lease_token: str) -> Awaitable[DrainCheck]: ...
    def set_waiter(self, lane: str, waiter: str) -> Awaitable[bool]:
        """``False`` when the lease is free (acquire instead)."""

    def waiter(self, lane: str) -> Awaitable[str | None]: ...
    def pickup_task(self, lane: str, task_id: str) -> Awaitable[str | None]:
        """The status the row had (``working``: a dead holder had it), or ``None`` when it was
        not runnable."""

    def mark_orphan_failed(
        self, lane: str, task_id: str, status_message: Any
    ) -> Awaitable[bool]: ...
    def write_items(
        self, lane: str, fence: Fence, items: list[ItemWrite] | list[dict[str, Any]]
    ) -> Awaitable[None]: ...
    def request_cancel(
        self, lane: str, task_id: str, reason: str | None = None
    ) -> Awaitable[bool]: ...
    def events(self, after_id: int = 0, limit: int = 100) -> Awaitable[list[EventRow]]: ...
    def lane_events(self, lane: str, after_id: int = 0) -> Awaitable[list[EventRow]]: ...
    def lanes(self) -> Awaitable[list[LaneRow]]: ...
    def lane(self, lane: str) -> Awaitable[LaneRow | None]: ...
    def history(self, lane: str) -> Awaitable[list[ChatItemRow]]: ...
    def tasks(
        self, lane: str, status: str | None = None, limit: int = 50
    ) -> Awaitable[list[TaskRow]]:
        """Newest first."""

    def orphans(self, lane: str) -> Awaitable[list[TaskRow]]: ...
    def inbox(self, lane: str) -> Awaitable[list[InboxRow]]: ...
    def inbox_pending(self, lane: str) -> Awaitable[list[InboxRow]]: ...

class Lease:
    """A held lease with its renewal timer (plan 3.3.2)."""

    @property
    def lane(self) -> str: ...
    @property
    def holder(self) -> str: ...
    @property
    def token(self) -> str: ...
    @property
    def is_lost(self) -> bool: ...
    def lost(self) -> Awaitable[None]:
        """Pending while the lease is held; raises ``StorageUnavailable`` once it is lost."""

    def stop(self) -> None: ...

class Streams:
    """The answer streams (plan 6) over ``/v1/streams``."""

    def create(
        self, ttl_seconds: int | None = None, region: str | None = None
    ) -> Awaitable[StreamInfo]: ...
    def append(self, stream_id: str, record: bytes) -> None:
        """Queued; sent in order, a failure is logged and does not fail the turn."""

    def append_now(self, stream_id: str, record: bytes) -> Awaitable[str]:
        """One append, awaited: the record's offset token."""

    def flush(self, stream_id: str) -> Awaitable[None]: ...
    def close(self, stream_id: str) -> Awaitable[None]: ...
    def delete(self, stream_id: str) -> Awaitable[None]: ...
    def read(self, stream_id: str, after: str | None = None, live: bool = False) -> StreamReader:
        """From ``after`` (an offset token; ``None`` is the beginning); ``live`` follows the tail
        until the stream closes."""

class StreamReader:
    """A read position over one stream."""

    def next(self) -> Awaitable[bytes | None]:
        """The next record, or ``None``: the tail of a catch-up read, the close of a live one."""

    def next_offset(self) -> Awaitable[str]: ...
    def is_closed(self) -> Awaitable[bool]: ...

class Threads:
    """The threads service's client (plan 4.3)."""

    def create_thread(
        self,
        *,
        scope: str = "",
        attributes: dict[str, str] | None = None,
        idle_ttl_seconds: int = 0,
    ) -> Awaitable[ThreadInfo]:
        """A new thread in ``scope``, with its database."""

    def get_thread(self, thread_id: str) -> Awaitable[ThreadInfo]: ...
    def list_threads(
        self, scope: str, *, before: str = "", limit: int = 0
    ) -> Awaitable[ThreadPage]:
        """The threads of one scope, newest first."""

    def update_thread(
        self,
        thread_id: str,
        *,
        attributes: dict[str, str] | None = None,
        idle_ttl_seconds: int | None = None,
    ) -> Awaitable[ThreadInfo]:
        """Merges ``attributes`` (an empty value deletes the entry) and sets the idle TTL when
        given."""

    def delete_thread(self, thread_id: str) -> Awaitable[None]: ...
    def get_thread_items(
        self, thread_id: str, *, agent: str = "", after: str = "", limit: int = 0
    ) -> Awaitable[ThreadItemPage]:
        """The thread's chat items, oldest first; ``agent`` keeps one agent's."""

    def get_thread_tasks(
        self, thread_id: str, *, status: str | None = None
    ) -> Awaitable[list[ThreadTask]]:
        """The thread's background tasks, oldest first, of one ``status`` when given."""

class StorageError(RuntimeError):
    """A session storage failure (plan 5.2)."""

class ThreadNotFound(StorageError):
    """An id the threads service does not know, or a database agent-db does not have."""

class VersionConflict(StorageError):
    """A fenced commit matched no row: the lane moved past the version the run loaded."""

class NotSerializable(StorageError):
    """A field that could not be serialized; the message names it."""

class TooLarge(StorageError):
    """A value over its size limit; the message names what and the limit."""

class StorageUnavailable(StorageError):
    """Retryable: the connection was lost, the lease lapsed, or a transaction was discarded."""

class RunRejected(StorageError):
    """The ``reject`` policy refused the message while a turn runs."""

    running_task_id: str | None

# --- turn.rs -------------------------------------------------------------------------------------

class Thread:
    """The run module's handle over an open database (plan 5.2): leases and waiter marks carry
    ``holder`` plus a handle id of their own."""

    def __init__(self, db: ThreadDb, holder: str) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def holder(self) -> str: ...
    @property
    def now_ms(self) -> int: ...
    def lanes(self) -> Awaitable[list[LaneInfo]]: ...
    def events(self, after_id: int = 0, limit: int = 100) -> Awaitable[list[EventRow]]: ...
    def history(self, lane: str) -> Awaitable[list[Item]]:
        """A lane's history as stored, in order."""

    def tasks(self, lane: str, limit: int = 50) -> Awaitable[list[TaskRecord]]:
        """A lane's task rows as records, newest first."""

    def close(self) -> None: ...

class Boundary:
    """The handler's boundary: ask before each model call and once more before the answer
    streams; ``True`` once means ``True`` from then on."""

    @property
    def task_id(self) -> str: ...
    @property
    def tripped(self) -> bool: ...
    @property
    def client_canceled(self) -> bool: ...
    def check(self, read: bool = False) -> Awaitable[bool]:
        """Whether to yield now; with ``read`` it reads even right after the turn took its
        batch, because the caller was told that something landed."""
    def record(self, item: Any, message_id: str | None = None) -> Awaitable[None]: ...

class TurnHostProtocol(Protocol):
    """What ``Turn`` calls back into: the handler and the wire, one object (plan 13.8).

    ``handle`` receives the ``storage_turn.TurnContext`` model (``_core.TurnContext`` is the
    turn detector's class of the same name) and the ``Boundary`` it polls, and returns a
    ``HandlerOutput`` (a dict in its shape is accepted too). An exception it raises is the
    handler's error, ``"<TypeName>: <text>"`` on the failed row; ``asyncio.CancelledError`` ends
    the run ``canceled``. The coroutines run on the loop that constructed the ``Turn`` and are
    cancelled when Rust drops the operation. Under ``live`` the holder also calls, when the
    object has them, ``on_end(record, output)`` as a task's turn commits and
    ``background(wait, cancel)`` between turns (``storage_turn.Background``) and ``discard(task_id)``
    for the work of a task that ended before it. Every request calls, when the object has it,
    ``accepted(task_id, resent, holds)`` once the turn knows which task the message is (its own,
    the question it continues, or for a resend the task its message id made) and whether it took
    the lane. A request that waits on another holder calls, when the object has them,
    ``arrived(task_id)`` once its message is in the inbox and ``settled(task_id, wait) -> bool``
    between reads of its task's row.
    """

    def handle(
        self, ctx: storage_turn.TurnContext, boundary: Boundary
    ) -> Awaitable[HandlerOutput]: ...
    def on_update(self, task_id: str, text: str) -> Awaitable[None]: ...
    def on_token(self, task_id: str, delta: str) -> Awaitable[None]: ...

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
