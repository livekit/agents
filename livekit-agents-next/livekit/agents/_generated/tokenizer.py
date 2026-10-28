# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tokenize/bindings.rs

from __future__ import annotations

from typing import TypeAlias

TokenizerToken: TypeAlias = str | tuple[str, int, int]
"""A tokenizer may supply text or character-indexed spans."""
