# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/remote.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .tool_report import ToolOutcome


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteEventPending:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteEventProgress:
    text: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteEventInputRequired:
    question: str
    request_id: str
    verbatim: str | None
    verbatim_full: bool
    outcome: ToolOutcome


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteEventOutcome:
    outcome: ToolOutcome


@dataclass(frozen=True, kw_only=True, slots=True)
class RemoteEventFollow:
    task_id: str
    thread_id: str


RemoteEvent: TypeAlias = (
    RemoteEventPending
    | RemoteEventProgress
    | RemoteEventInputRequired
    | RemoteEventOutcome
    | RemoteEventFollow
)
