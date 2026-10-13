# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/threads.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ThreadInfo(BaseModel):
    """A thread as the host reads it (plan 4.3): the proto's fields, times as unix seconds."""

    model_config = ConfigDict(frozen=True)
    thread_id: str
    scope: str
    """The subject the thread belongs to, the one a grant is scoped by."""
    attributes: dict[str, str]
    created_at: int
    last_active: int
    expires_at: int
    """Unix seconds; `0` for never."""
    idle_ttl_seconds: int
    database_id: str
    """The thread's database on agent-db, `DB_..`, created with the thread."""


class ThreadPage(BaseModel):
    """One page of `ListThreads`, newest first."""

    model_config = ConfigDict(frozen=True)
    threads: list[ThreadInfo]
    before: str
    """The cursor for the next page; empty at the end."""


class ThreadItem(BaseModel):
    """One chat item of a thread as the host reads it; `item` is the `ChatContext.ChatItem` as
    protobuf JSON.
    """

    model_config = ConfigDict(frozen=True)
    agent: str
    seq: int
    """Position in the agent's history."""
    message_id: str
    item: Any


class ThreadItemPage(BaseModel):
    """One page of `GetThreadItems`, oldest first."""

    model_config = ConfigDict(frozen=True)
    items: list[ThreadItem]
    after: str
    """The cursor for the next page; empty at the end."""


class ThreadTask(BaseModel):
    """One background task of a thread, as `GetThreadTasks` answers it."""

    model_config = ConfigDict(frozen=True)
    task_id: str
    thread_id: str
    name: str
    status: str
    """`running`, `completed`, `failed` or `canceled`."""
    updated_at: int
