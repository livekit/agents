# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/storage/session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from livekit.agents.storage.context import ContextStore

Context_T = TypeVar("Context_T")


SessionKind: TypeAlias = Literal["empty", "conversation"]


@dataclass(frozen=True)
class BoundContext(Generic[Context_T]):
    """An inert execution declaration; tools receive the original context object."""

    context: Context_T
    store: ContextStore


BoundContext.__module__ = "livekit.agents.storage.session_state"
