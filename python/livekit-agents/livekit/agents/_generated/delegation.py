# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/delegation/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations

from livekit.agents.llm import LLM


@dataclass(frozen=True, kw_only=True, slots=True)
class DelegationOptions:
    """Conversation-wide delegation behavior, shared by every remote tool on an execution owner."""

    routing_model: LLM | None = None
    """Separate routing model; None disables automatic routing and requires explicit task selection."""


DelegationOptions.__annotations__ = get_annotations(DelegationOptions, eval_str=True)
DelegationOptions.__module__ = "livekit.agents"
