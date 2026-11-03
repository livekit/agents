# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/turn.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .chat_items import ChatItem
    from .storage_turn import PendingCall
    from .usage import Usage


@dataclass(kw_only=True, slots=True)
class HandlerOutcomeValue:
    value: object


@dataclass(kw_only=True, slots=True)
class HandlerOutcomeCompleted:
    value: object
    verbatim: bool


@dataclass(kw_only=True, slots=True)
class HandlerOutcomeInputRequired:
    question: str


@dataclass(kw_only=True, slots=True)
class HandlerOutcomeFailed:
    error: str


@dataclass(kw_only=True, slots=True)
class HandlerOutcomeRejected:
    reason: str


@dataclass(kw_only=True, slots=True)
class HandlerOutcomeWorking:
    message: str | None


HandlerOutcome: TypeAlias = (
    HandlerOutcomeValue
    | HandlerOutcomeCompleted
    | HandlerOutcomeInputRequired
    | HandlerOutcomeFailed
    | HandlerOutcomeRejected
    | HandlerOutcomeWorking
)


@dataclass(kw_only=True, slots=True)
class HandlerRun:
    items: list[ChatItem]
    usage: Usage
    verbatim: str | None
    question: str | None
    pending_call: PendingCall | None
