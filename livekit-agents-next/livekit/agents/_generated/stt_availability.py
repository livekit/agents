# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/stt/stt.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations

from .stt import STT


@dataclass
class AvailabilityChangedEvent:
    stt: STT
    available: bool


AvailabilityChangedEvent.__annotations__ = get_annotations(AvailabilityChangedEvent, eval_str=True)
AvailabilityChangedEvent.__module__ = "livekit.agents.stt"
