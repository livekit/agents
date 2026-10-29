"""Stubs of the ``livekit.agents._core`` names whose pyo3 code is still handwritten.

``_core.pyi`` (generated) re-exports exactly the names this file defines that
``lk-agents-codegen``'s ``surface::HANDWRITTEN`` lists; ``core_names.json`` checks them against the
extension's registrations. A family whose pyo3 code moves to ABI declarations leaves both, and its
stub is then generated. The ``*Protocol`` classes describe host objects and are not ``_core`` names.
"""

from __future__ import annotations

from collections.abc import Awaitable
from enum import Enum
from typing import Any, Protocol

from ._generated.storage_class import (
    EventRow,
    InboxRow,
    ItemRow,
    LaneRow,
    StreamInfo,
    TaskPage,
    TaskRecord,
    ThreadItemPage,
)
from ._generated.storage_context import ContextScope, ContextScopeWrite
from ._generated.storage_options import Policy, TurnOptions
from ._generated.storage_turn import TurnContext, TurnResult

# --- runtime.rs ----------------------------------------------------------------------------------

class Role(Enum):
    Worker = ...
    Job = ...

def init(role: Role) -> None: ...

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
        lane: str | None = None,
        status: str | None = None,
        order: str = "updated",
        page_size: int = 50,
        page_token: str | None = None,
    ) -> Awaitable[TaskPage]:
        """One page of the thread's tasks: ``"updated"``, last updated first (A2A
        ``tasks/list``), or ``"inserted"``, oldest first; of ``lane`` and ``status`` when given."""

    def list_items(
        self, lane: str | None = None, after: str | None = None, limit: int = 100
    ) -> Awaitable[ThreadItemPage]:
        """One page of the thread's committed items in insertion order, after a page's ``after``."""

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

def resolve_host_ref(id: int) -> Any:
    """The object the host interned under ``id``; read, not released."""

# --- tokenize ------------------------------------------------------------------------------------

# --- inference -----------------------------------------------------------------------------------
