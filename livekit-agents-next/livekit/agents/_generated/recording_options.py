# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/telemetry/report/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing_extensions import TypedDict


class RecordingOptions(TypedDict, total=False):
    """Which recording features are active; keys default to ``True`` when not specified."""

    audio: bool
    traces: bool
    logs: bool
    transcript: bool
    redaction: bool


RecordingOptions.__module__ = "livekit.agents.voice._session_bindings"
