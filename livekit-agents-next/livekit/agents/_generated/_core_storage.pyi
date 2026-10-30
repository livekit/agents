# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from .._core import Thread
from .chat_items import ChatItem
from .storage_class import TaskState, ThreadInfo, ThreadItemPage, ThreadPage, ThreadTask
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
        directory: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: float | None = None,
        renew_interval: float | None = None,
    ) -> Self:
        """`directory`: threads in that local directory, resolved to an absolute path once.
        Otherwise agentstore through the public API derived from `url` (the project's
        `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
        """
    @property
    def location(self) -> str:
        """The absolute local directory, or the public API agentstore is reached through."""
    async def create_thread(
        self,
        *,
        scope: str | None = None,
        attributes: dict[str, object] | None = None,
        idle_ttl_seconds: int = 0,
    ) -> ThreadInfo:
        """A new `AT_` thread, returned once its metadata and database exist. `attributes` maps
        strings to strings.
        """
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
        attributes: dict[str, object] | None = None,
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

class _core_VoiceStore:
    """The stable context store and initial snapshot belonging to one live call's lane."""
    @property
    def initial_history(self) -> list[ChatItem] | None:
        """`None` preserves the application's explicit constructor history."""
    @property
    def activation(self) -> _core_ContextActivation: ...
    async def aclose(self) -> None: ...
