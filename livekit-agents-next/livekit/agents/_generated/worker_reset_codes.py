# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/worker/endpoint/reset_codes.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Final

HSR_ABORT: Final = 0

HSR_REFUSED: Final = 1

HSR_INTERNAL: Final = 2

HSR_TIMEOUT: Final = 3

HSR_PROTOCOL: Final = 4
