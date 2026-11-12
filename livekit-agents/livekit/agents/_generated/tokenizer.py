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


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A token and its start and end offsets, in characters, in the text it was cut from: what a
    tokenizer's `split` gives a stream to batch, and what a supplied tokenizer may return.
    """

    token: str
    start: int
    end: int


@dataclass
class ReplacedWords:
    """`replace_words` over a chunk: the text with its replacements, and how much of it is final
    (`processed`, in characters); the rest may still change with the next chunk.
    """

    text: str
    processed: int


TokenizerToken: TypeAlias = str | TokenSpan
"""A tokenizer may supply text or character-indexed spans."""


TokenData.__annotations__ = get_annotations(TokenData, eval_str=True)
TokenData.__module__ = "livekit.agents.tokenize"
TokenSpan.__annotations__ = get_annotations(TokenSpan, eval_str=True)
TokenSpan.__module__ = "livekit.agents.tokenize"
