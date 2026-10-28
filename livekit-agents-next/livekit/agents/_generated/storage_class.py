# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/storage/mod.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class ThreadInfo:
    """A thread: times are unix seconds, `0` for never, TTLs whole seconds."""

    thread_id: str
    scope: str
    """The subject the thread is about, the one threads are listed and granted by."""
    attributes: dict[str, object]
    created_at: int
    last_active: int
    expires_at: int
    idle_ttl_seconds: int
    database_id: str
    """The thread's database: `DB_..` on agent-db, the thread id locally."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ThreadPage:
    """One page of `list_threads`, newest first."""

    threads: list[ThreadInfo]
    before: str
    """The next page's cursor; empty at the end."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ThreadItem:
    """One chat item of a thread; `item` is the `ChatContext.ChatItem` as protobuf JSON."""

    agent: str
    seq: int
    """Position in the thread's history."""
    message_id: str
    item: object


@dataclass(frozen=True, kw_only=True, slots=True)
class ThreadItemPage:
    """One page of `get_thread_items`, oldest first."""

    items: list[ThreadItem]
    after: str
    """The next page's cursor; empty at the end."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ThreadTask:
    """One A2A task of a thread."""

    task_id: str
    thread_id: str
    name: str
    """The application's task label; empty when none."""
    status: str
    """The A2A state: `submitted`, `working`, `input-required`, `completed`, ..."""
    updated_at: int
