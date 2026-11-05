# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/models.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal, TypeAlias

VADModels: TypeAlias = Literal["silero"]


TurnDetectorModels: TypeAlias = Literal["turn-detector-v1", "turn-detector-v1-mini"]


TurnDetectorVersions: TypeAlias = Literal["v1", "v1-mini"]


InferenceClass: TypeAlias = Literal["priority", "standard", "low"]
"""Scheduling class for a request. ``low`` yields to voice traffic, so it is only
appropriate for work no caller is waiting on.
"""
