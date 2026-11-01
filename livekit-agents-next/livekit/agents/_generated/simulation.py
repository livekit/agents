# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass
class SimulationVerdict:
    """A pass/fail verdict for a scenario, with a human-readable reason."""

    success: bool
    reason: str


SimulationVerdict.__annotations__ = get_annotations(SimulationVerdict, eval_str=True)
SimulationVerdict.__module__ = "livekit.agents.simulation"
