# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/chat_context/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass
class DiffOps:
    to_remove: list[str]
    to_create: list[tuple[str | None, str]]
    to_update: list[tuple[str | None, str]]


DiffOps.__annotations__ = get_annotations(DiffOps, eval_str=True)
DiffOps.__module__ = "livekit.agents.llm.utils"
