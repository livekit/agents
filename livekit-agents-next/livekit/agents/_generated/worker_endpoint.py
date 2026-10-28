# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/endpoint.rs

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointRequest:
    """The authenticated node preamble attached to an ASGI request."""

    request_id: str
    authorized: bool
    route: str
    kind: str
    client_addr: str
    scheme: str
    timeout_ms: int


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointOptions:
    """Application HTTP route visibility; A2A routes always require authentication."""

    public: bool = False
