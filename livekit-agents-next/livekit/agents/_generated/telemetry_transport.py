# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/telemetry/transport.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class TraceCarrier:
    traceparent: str | None = None
