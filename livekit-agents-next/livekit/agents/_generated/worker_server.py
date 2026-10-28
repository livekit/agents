# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/worker.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class RegisteredInfo:
    """Registration accepted by the server, delivered as a typed record."""

    worker_id: str
    """The id the server issued."""
    region: str
    protocol: int
    """The protocol version the server chose."""
    deployment: str
    agent_name: str
