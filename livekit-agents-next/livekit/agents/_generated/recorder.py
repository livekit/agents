# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/recorder.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class RecordingSummary:
    bytes_written: int
    started_at: float | None
    last_flushed: float | None
    dropped_input_samples: int
    dropped_output_samples: int
