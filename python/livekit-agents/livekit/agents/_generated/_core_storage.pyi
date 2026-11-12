# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Self

from livekit.protocol.agent import Job

from .._core import _core_HandlerTurn
from .chat_items import ChatItem
from .storage_class import (
    EventRow,
    InboxRow,
    ItemRow,
    LocalStorageOptions,
    PartitionRow,
    RunOutcome,
    TaskOrder,
    TaskPage,
    TaskRecord,
    TaskState,
    ThreadInfo,
    ThreadItemPage,
    ThreadPage,
    ThreadTask,
)
from .storage_context import (
    AppStatement,
    ContextField,
    ContextScope,
    ContextScopeWrite,
    ContextValues,
)
from .storage_session import BoundContext, SessionKind
from .text_session import FrameId, SessionFrame, TextSessionState

class Storage:
    """Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
    directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
    """
    async def _core_serve_threads(
        self,
        api_key: str,
        api_secret: str,
        port: int = 0,
    ) -> _core_ThreadServer: ...
    def __new__(
        cls,
        *,
        local: LocalStorageOptions | None = None,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str | None = None,
    ) -> Self:
        """`local`: threads in that project's directory, resolved to an absolute path once.
        Otherwise agentstore through the public API derived from `url`, with tokens minted from
        the key pair; each omitted one is `LIVEKIT_URL`, `LIVEKIT_API_KEY` or
        `LIVEKIT_API_SECRET`, and an omitted lease timing `LIVEKIT_AGENTDB_LEASE_TTL` or
        `LIVEKIT_AGENTDB_RENEW_INTERVAL` (float seconds). An omitted or empty `holder` is the
        constructing process's `<hostname>-<pid>`. Opens nothing until first use.
        """
    @property
    def location(self) -> str:
        """The absolute local directory, or the public API agentstore is reached through."""
    @property
    def holder(self) -> str:
        """This process's lease holder id."""
    @property
    def project_id(self) -> str: ...
    async def create_thread(
        self,
        *,
        scope: str | None = None,
        attributes: Mapping[str, str] | None = None,
        idle_ttl_seconds: int = 0,
    ) -> ThreadInfo:
        """A new `AT_` thread, returned once its metadata and database exist."""
    async def get_thread(self, thread_id: str) -> ThreadInfo:
        """Unknown, deleted or expired threads raise `ThreadNotFound`."""
    async def list_threads(
        self,
        scope: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> ThreadPage:
        """The threads of one scope, newest first; `before` is the previous page's cursor."""
    async def update_thread(
        self,
        thread_id: str,
        *,
        attributes: Mapping[str, str] | None = None,
        idle_ttl_seconds: int | None = None,
    ) -> ThreadInfo:
        """Merges `attributes` (an empty value deletes the key) and sets the idle TTL when given:
        the activity boundary that moves expiry.
        """
    async def delete_thread(self, thread_id: str) -> None:
        """Makes the thread unavailable and reclaims its content; repeat-safe."""
    async def get_thread_items(
        self,
        thread_id: str,
        *,
        agent: str | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> ThreadItemPage:
        """The thread's chat items, oldest first, of one agent's partition when `agent` is given."""
    async def get_thread_tasks(
        self,
        thread_id: str,
        *,
        status: TaskState | None = None,
    ) -> list[ThreadTask]:
        """The thread's A2A tasks, of one `status` when given."""
    async def _core_call_thread(self, job: Job) -> ThreadInfo | None:
        """A call's thread: the one the job's `thread_id` names, `None` when it names none. Never
        creates a thread.
        """
    async def open(self, thread_id: str) -> Thread:
        """The existing thread's handle. Takes no lease."""
    async def _core_context_store(
        self,
        thread_id: str,
        partition_key: str,
    ) -> _core_ContextActivation:
        """A runtime-supplied field store whose explicit saves commit outside a managed turn."""
    async def _core_turn_store(
        self,
        thread_id: str,
        partition_key: str,
        scopes: Sequence[ContextScope],
        *,
        routing_state: object | None = None,
        read_only: bool = False,
    ) -> _core_ContextActivation:
        """A managed turn's field store over the scopes and routing state its run loaded: saves
        and `transaction()` writes stage for the turn's commit.
        """
    async def aclose(self) -> None:
        """Releases backend resources once open handles settled; repeat-safe."""

class _core_ContextActivation:
    """Activation state and revision/schema decisions. Application type validation stays in the host."""
    def __new__(
        cls,
        scopes: Sequence[ContextScope],
        routing_state: object | None,
        read_only: bool,
    ) -> Self: ...
    async def load(
        self,
        scope_id: str,
        identity: str,
        fields: Sequence[ContextField],
    ) -> ContextValues:
        """Check manifests before restoring values or accepting defaults for absent fields."""
    async def save(
        self,
        scope_id: str,
        identity: str,
        revision: int,
        fields: Sequence[ContextField],
    ) -> ContextValues:
        """Stage the validated fields; a stale activation cannot overwrite a newer revision."""
    async def load_session(self) -> object | None:
        """A detached read of committed routing state; never starts an execution."""
    async def load_session_snapshot(self) -> str: ...
    def declare_binding(self, context: object, store: object) -> BoundContext: ...
    async def flush_bound_session(self, history: Sequence[ChatItem] | None) -> None: ...
    def set_session_sink(self, sink: _core_HandlerTurn) -> None:
        """The managed turn's checkpoint channel, owned by the run holding the lease."""
    async def flush_session(self, history: Sequence[ChatItem]) -> None:
        """Persist the previously accepted session state, without saving application fields."""
    def stage_session(self, state: object) -> None:
        """Stage framework-owned session state without saving ordinary application fields."""
    @property
    def session_state(self) -> object | None: ...
    @property
    def writes(self) -> list[ContextScopeWrite]: ...
    @property
    def statements(self) -> list[AppStatement]:
        """The application statements managed transactions staged for the turn's commit."""
    async def transaction(self, independent: bool) -> _core_AppTransaction:
        """Application SQL (docs/context.md#application-records-and-transactions): `independent`
        commits when the transaction ends; otherwise it joins the managed turn's commit, or
        commits like an independent one outside a managed turn.
        """
    def acknowledge(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None:
        """End the activation, refusing subsequent loads or saves."""

class _core_AppTransaction:
    """One application transaction. Reads observe committed records; writes are staged and
    applied atomically when the transaction ends, so a read after a write in the same
    transaction is refused.
    """
    async def execute(self, sql: str, params: Sequence[object] | None = None) -> None:
        """Stages one statement; `params` are JSON scalars."""
    async def fetch(self, sql: str, params: Sequence[object] | None = None) -> list[list[object]]:
        """One read's rows, each a list of column values in select order."""
    async def fetch_one(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> list[object] | None:
        """The first row of one read, `None` when it has none."""
    async def commit(self) -> None:
        """Ends the transaction: its writes commit (independent) or join the turn's commit."""
    def rollback(self) -> None:
        """Ends the transaction, discarding its writes."""

class Thread:
    @property
    def id(self) -> str: ...
    async def partitions(self) -> list[PartitionRow]: ...
    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]: ...
    async def history(self, partition_key: str) -> list[ItemRow]: ...
    async def tasks(self, partition_key: str, limit: int = 50) -> list[TaskRecord]: ...
    async def task(self, task_id: str) -> TaskRecord | None: ...
    async def list_tasks(
        self,
        partition_key: str | None = None,
        status: TaskState | None = None,
        order: TaskOrder = "updated",
        page_size: int = 50,
        page_token: str | None = None,
    ) -> TaskPage: ...
    async def list_items(
        self,
        partition_key: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> ThreadItemPage: ...
    async def partition_info(self, partition_key: str) -> PartitionRow | None: ...
    async def inbox(self, partition_key: str, pending: bool = False) -> list[InboxRow]: ...
    async def cancel_task(
        self,
        partition_key: str,
        task_id: str,
        metadata: dict[str, object],
    ) -> TaskRecord | None: ...
    async def open_run(
        self,
        partition_key: str,
        run_id: str,
        *,
        channel: str | None = None,
        wait_timeout: timedelta | None = None,
    ) -> Run: ...
    def close(self) -> None:
        """Schedule connection release without blocking the caller."""

class Run:
    @property
    def version(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    @property
    def history(self) -> list[ItemRow]: ...
    @property
    def state(self) -> object | None: ...
    @property
    def context_scopes(self) -> list[ContextScope]: ...
    async def commit(
        self,
        history: Sequence[ItemRow],
        state: object | None = None,
        *,
        scope_writes: Sequence[ContextScopeWrite] | None = None,
    ) -> int: ...
    async def abandon(self, outcome: RunOutcome, reason: str) -> None: ...

class _core_ThreadHandle:
    """`ctx.thread` (docs/threads.md): one thread of a storage namespace, as the partition_key `partition_key` reads
    it. Each read opens the thread's database for the call and releases it after; there is
    nothing to open or close.
    """
    def __new__(cls, namespace: Storage, thread_id: str, partition_key: str) -> Self:
        """Reads take no lease."""
    @property
    def id(self) -> str:
        """The thread id."""
    @property
    def partition_key(self) -> str:
        """The partition_key whose history `chat_history()` reads by default."""
    @property
    def namespace(self) -> Storage:
        """The namespace the thread lives in."""
    async def info(self) -> ThreadInfo:
        """The platform's record: scope, attributes, created, last active, expiry."""
    async def update(
        self,
        *,
        attributes: Mapping[str, str] | None = None,
        idle_ttl_seconds: int | None = None,
    ) -> ThreadInfo:
        """Change the attributes a listing returns, or the idle period."""
    async def list_items(
        self,
        *,
        partition_key: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> ThreadItemPage:
        """One page of the thread's committed chat items in insertion order, every partition's or
        `partition_key`'s; pass the page's `after` to read the next (empty on the last page). The same
        records Cloud's threads API answers, for an application's own read routes.
        """
    async def list_tasks(
        self,
        *,
        partition_key: str | None = None,
        status: TaskState | None = None,
        order: TaskOrder = "inserted",
        page_token: str | None = None,
        limit: int = 50,
    ) -> TaskPage:
        """One page of the thread's A2A task records: in insertion order (`order="inserted"`), or
        last updated first (`"updated"`) as A2A `tasks/list` lists them; of `partition_key`'s partition and
        `status` when given. Pass the page's `next_page_token` to read the next; a token is bound
        to its listing.
        """
    async def task(self, task_id: str) -> TaskRecord | None:
        """One task record, `None` when this thread has none of that id."""
    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]:
        """One page of the journal (docs/internals/storage.md#journal), every partition, in order; pass
        the last `id` to read the next.
        """
    async def history(self, partition_key: str) -> list[ItemRow]:
        """`partition_key`'s partition history, in order."""
    async def tasks(
        self,
        partition_key: str,
        *,
        status: TaskState | None = None,
        limit: int = 50,
    ) -> list[TaskRecord]:
        """`partition_key`'s partition task records, most recent first; only those in `status` when given."""

class _core_ThreadServer:
    @property
    def port(self) -> int: ...
    async def aclose(self) -> None: ...

class _core_PersistedSessionState:
    """Read-only conversation state. Reads and results are detached; an explicitly bound
    application field retains its identity while the native store updates its contents.
    """
    def __new__(cls, data: str | None = None) -> Self: ...
    @property
    def kind(self) -> SessionKind: ...
    @property
    def active_agent(self) -> str | None: ...
    @property
    def session_id(self) -> str | None: ...
    @property
    def execution_id(self) -> str | None: ...
    @property
    def root_frame_id(self) -> FrameId: ...
    @property
    def active_frame_id(self) -> FrameId: ...
    @property
    def ancestors(self) -> list[FrameId]: ...
    @property
    def frames(self) -> dict[FrameId, SessionFrame]: ...
    def snapshot(self) -> str: ...
    def _to_json(self) -> str: ...
    def _text_state(self) -> TextSessionState | None: ...
    def _update(self, data: str) -> None: ...
    @staticmethod
    def _from_json(data: str) -> str: ...
    @staticmethod
    def _from_text(state: TextSessionState) -> str: ...
    @staticmethod
    def _from_voice(agent: str, history_ids: Sequence[str]) -> str: ...

class _core_SessionBinding:
    """Native invocation-local binding and checkpoint publisher."""
    def __new__(
        cls,
        activation: _core_ContextActivation,
        view: _core_PersistedSessionState,
    ) -> Self: ...
    def update(self, state: str) -> None: ...
    def update_voice(self, agent: str, history_ids: Sequence[str]) -> None: ...
    def validate_voice(self, agent: str) -> None: ...
    async def flush(self, history: Sequence[ChatItem] | None = None) -> None: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...

class _core_VoiceStore:
    """The stable context store and initial snapshot belonging to one live call's partition."""
    @property
    def initial_history(self) -> list[ChatItem] | None:
        """`None` preserves the application's explicit constructor history."""
    @property
    def activation(self) -> _core_ContextActivation: ...
    async def aclose(self) -> None: ...
