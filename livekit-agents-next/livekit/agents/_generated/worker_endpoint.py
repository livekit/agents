# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/http/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

ExchangeKind: TypeAlias = Literal["http", "websocket"]
"""Whether the exchange is an HTTP request or a WebSocket upgrade."""


@dataclass(kw_only=True, slots=True)
class RequestHead:
    """The parsed request head. `path` is percent-decoded; `raw_path` and `query_string` are the
    bytes on the wire.
    """

    kind: ExchangeKind
    method: str
    http_version: str
    path: str
    raw_path: bytes
    query_string: bytes
    headers: list[tuple[bytes, bytes]]
    subprotocols: list[str]
    """The WebSocket subprotocols the client offered."""


ExchangeEvent: TypeAlias = MutableMapping[str, Any]


ExchangeMessage: TypeAlias = MutableMapping[str, Any]


EndpointKind: TypeAlias = Literal["http", "text"]
"""The payload semantics declared in the Cloud route manifest and node preamble."""


HttpScheme: TypeAlias = Literal["http", "https"]
"""The end client's scheme, asserted by the node rather than inferred from worker transport."""


@dataclass(kw_only=True, slots=True)
class EndpointRequest:
    """The authenticated node preamble attached to an endpoint byte stream."""

    request_id: str
    authorized: bool
    route: str
    kind: EndpointKind
    client_addr: str
    scheme: HttpScheme
    timeout_ms: int
    unauthenticated: bool = False
    """The console's listener: nothing authenticates the caller, who holds every permission."""


@dataclass(kw_only=True, slots=True)
class EndpointOptions:
    """Application HTTP route visibility; A2A routes always require authentication."""

    public: bool = False


EndpointRequest.__module__ = "livekit.agents.http"
