# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/mod.rs
#     rust/crates/lk-agents-abi/src/tokenize/bindings.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import TypeAlias


@dataclass
class TokenData:
    segment_id: str = ""
    token: str = ""


TokenizerToken: TypeAlias = str | tuple[str, int, int]
"""A tokenizer may supply text or character-indexed spans."""


TokenData.__annotations__ = get_annotations(TokenData, eval_str=True)
TokenData.__module__ = "livekit.agents.tokenize"
