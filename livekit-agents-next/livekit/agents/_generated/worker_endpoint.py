# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/http.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

EndpointKind: TypeAlias = Literal["http", "text"]
"""The payload semantics declared in the Cloud route manifest and node preamble."""


HttpScheme: TypeAlias = Literal["http", "https"]
"""The end client's scheme, asserted by the node rather than inferred from worker transport."""


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointRequest:
    """The authenticated node preamble attached to an endpoint byte stream."""

    request_id: str
    authorized: bool
    route: str
    kind: EndpointKind
    client_addr: str
    scheme: HttpScheme
    timeout_ms: int


@dataclass(frozen=True, kw_only=True, slots=True)
class EndpointOptions:
    """Application HTTP route visibility; A2A routes always require authentication."""

    public: bool = False
