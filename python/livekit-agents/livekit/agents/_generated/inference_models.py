# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/mod.rs
#     rust/crates/lk-agents-abi/src/inference/models.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .turn_events import TurnDetectorModel


@dataclass(kw_only=True, slots=True)
class EotSelection:
    """The model a turn detector runs, and why it is not the one asked for, to log."""

    model: TurnDetectorModel
    warnings: list[str]


VADModels: TypeAlias = Literal["silero"]


TurnDetectorModels: TypeAlias = Literal["turn-detector-v1", "turn-detector-v1-mini"]


TurnDetectorVersions: TypeAlias = Literal["v1", "v1-mini"]


InferenceClass: TypeAlias = Literal["priority", "standard", "low"]
"""Scheduling class for a request. ``low`` yields to voice traffic, so it is only
appropriate for work no caller is waiting on.
"""
