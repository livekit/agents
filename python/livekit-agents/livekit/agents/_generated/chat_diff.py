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
class AnchoredItem:
    """An item to create or patch, placed after `previous_item_id` (`None`: first in the
    conversation).
    """

    previous_item_id: str | None
    item_id: str


@dataclass
class DiffOps:
    """The operations turning one conversation into another; an update patches the item in place,
    keeping its server-side position.
    """

    to_remove: list[str]
    to_create: list[AnchoredItem]
    to_update: list[AnchoredItem]


@dataclass
class MergeInsertion:
    """`ChatContext.merge`'s insertion: `theirs[item]` goes in at `index`."""

    index: int
    item: int


AnchoredItem.__annotations__ = get_annotations(AnchoredItem, eval_str=True)
AnchoredItem.__module__ = "livekit.agents.llm.utils"
DiffOps.__annotations__ = get_annotations(DiffOps, eval_str=True)
DiffOps.__module__ = "livekit.agents.llm.utils"
