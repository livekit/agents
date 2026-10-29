# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/server_host.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointRoute:
    """One route of the endpoint manifest the worker registers."""

    path: str
    methods: list[str]
    kind: str
    """`"http"`, or `"text"` for a session endpoint."""
    public: bool
