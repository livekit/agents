# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from .chat_items import ChatItem
from .storage_class import (
    EventRow,
    InboxRow,
    ItemRow,
    LaneRow,
    LocalStorageOptions,
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
    async def _core_voice_store(
        self,
        thread_id: str,
        lane: str,
        *,
        initial_history: Sequence[ChatItem] | None = None,
    ) -> _core_VoiceStore: ...
    def __new__(
        cls,
        *,
        local: LocalStorageOptions | None = None,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: float | None = None,
        renew_interval: float | None = None,
    ) -> Self:
        """`local`: threads in that project's directory, resolved to an absolute path once.
        Otherwise agentstore through the public API derived from `url` (the project's
        `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
        """
    @property
    def location(self) -> str:
        """The absolute local directory, or the public API agentstore is reached through."""
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
        """The thread's chat items, oldest first, of one agent's lane when `agent` is given."""
    async def get_thread_tasks(
        self,
        thread_id: str,
        *,
        status: TaskState | None = None,
    ) -> list[ThreadTask]:
        """The thread's A2A tasks, of one `status` when given."""
    async def open(self, thread_id: str, holder: str) -> Thread:
        """The existing thread's handle, its writes under `holder`. Takes no lease."""
    async def _core_context_store(self, thread_id: str, lane: str) -> _core_ContextActivation:
        """A runtime-supplied field store whose explicit saves commit outside a managed turn."""
    async def _core_turn_store(
        self,
        thread_id: str,
        lane: str,
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
    async def execute(self, sql: str, params: Sequence[object]) -> None:
        """Stages one statement; `params` are JSON scalars."""
    async def fetch(self, sql: str, params: Sequence[object]) -> list[list[object]]:
        """One read's rows, each a list of column values in select order."""
    async def commit(self) -> None:
        """Ends the transaction: its writes commit (independent) or join the turn's commit."""
    def rollback(self) -> None:
        """Ends the transaction, discarding its writes."""

class Thread:
    @property
    def id(self) -> str: ...
    @property
    def holder(self) -> str: ...
    async def lanes(self) -> list[LaneRow]: ...
    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]: ...
    async def history(self, lane: str) -> list[ItemRow]: ...
    async def tasks(self, lane: str, limit: int = 50) -> list[TaskRecord]: ...
    async def task(self, task_id: str) -> TaskRecord | None: ...
    async def list_tasks(
        self,
        lane: str | None = None,
        status: TaskState | None = None,
        order: str = "updated",
        page_size: int = 50,
        page_token: str | None = None,
    ) -> TaskPage: ...
    async def list_items(
        self,
        lane: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> ThreadItemPage: ...
    async def lane_info(self, lane: str) -> LaneRow | None: ...
    async def inbox(self, lane: str, pending: bool = False) -> list[InboxRow]: ...
    async def cancel_task(
        self,
        lane: str,
        task_id: str,
        metadata: dict[str, object],
    ) -> TaskRecord | None: ...
    async def open_run(
        self,
        lane: str,
        run_id: str,
        *,
        channel: str | None = None,
        wait_timeout: float | None = None,
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
    async def abandon(self, outcome: str, reason: str) -> None: ...

class _core_ThreadHandle:
    """`ctx.thread` (docs/threads.md): one thread of a storage namespace, as the agent `agent` reads
    it. Each read opens the thread's database for the call and releases it after; there is
    nothing to open or close.
    """
    def __new__(cls, namespace: Storage, thread_id: str, holder: str, agent: str) -> Self:
        """`holder` signs the handle's writes; reads take no lease."""
    @property
    def id(self) -> str:
        """The thread id."""
    @property
    def agent(self) -> str:
        """The agent whose history `chat_history()` reads by default."""
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
        agent: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> ThreadItemPage:
        """One page of the thread's committed chat items in insertion order, every lane's or
        `agent`'s; pass the page's `after` to read the next (empty on the last page). The same
        records Cloud's threads API answers, for an application's own read routes.
        """
    async def list_tasks(
        self,
        *,
        agent: str | None = None,
        status: TaskState | None = None,
        order: str = "inserted",
        page_token: str | None = None,
        limit: int = 50,
    ) -> TaskPage:
        """One page of the thread's A2A task records: in insertion order (`order="inserted"`), or
        last updated first (`"updated"`) as A2A `tasks/list` lists them; of `agent`'s lane and
        `status` when given. Pass the page's `next_page_token` to read the next; a token is bound
        to its listing.
        """
    async def task(self, task_id: str) -> TaskRecord | None:
        """One task record, `None` when this thread has none of that id."""
    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]:
        """One page of the journal (docs/internals/storage.md#journal), every lane, in order; pass
        the last `id` to read the next.
        """
    async def history(self, agent: str) -> list[ItemRow]:
        """`agent`'s lane history, in order."""
    async def tasks(
        self,
        agent: str,
        *,
        status: TaskState | None = None,
        limit: int = 50,
    ) -> list[TaskRecord]:
        """`agent`'s lane task records, most recent first; only those in `status` when given."""

class _core_ThreadServer:
    @property
    def port(self) -> int: ...
    async def aclose(self) -> None: ...

class _core_VoiceStore:
    """The stable context store and initial snapshot belonging to one live call's lane."""
    @property
    def initial_history(self) -> list[ChatItem] | None:
        """`None` preserves the application's explicit constructor history."""
    @property
    def activation(self) -> _core_ContextActivation: ...
    async def aclose(self) -> None: ...
