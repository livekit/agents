# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/endpoint.rs
#     rust/crates/lk-agents-abi/src/worker/endpoint_host.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


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


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiScope:
    """The ASGI scope of one exchange, without `state` (the host adds `lk_request`)."""

    kind: str
    """`"http"` or `"websocket"`."""
    method: str
    """HTTP only."""
    scheme: str
    path: str
    raw_path: bytes
    query_string: bytes
    headers: list[tuple[bytes, bytes]]
    client: tuple[str, int] | None
    subprotocols: list[str]
    """WebSocket only."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiReceivedRequest:
    body: bytes
    more_body: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiReceivedDisconnect:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiReceivedWsConnect:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiReceivedWsText:
    text: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiReceivedWsBytes:
    data: bytes


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiReceivedWsDisconnect:
    code: int


AsgiReceived: TypeAlias = (
    AsgiReceivedRequest
    | AsgiReceivedDisconnect
    | AsgiReceivedWsConnect
    | AsgiReceivedWsText
    | AsgiReceivedWsBytes
    | AsgiReceivedWsDisconnect
)
"""What the application's `receive()` returns."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentResponseStart:
    status: int
    headers: list[tuple[bytes, bytes]]


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentResponseBody:
    body: bytes
    more_body: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentResponseTrailers:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentWsAccept:
    subprotocol: str | None
    headers: list[tuple[bytes, bytes]]


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentWsText:
    text: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentWsBytes:
    data: bytes


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentWsClose:
    code: int
    reason: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentWsHttpResponseStart:
    status: int
    headers: list[tuple[bytes, bytes]]


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentWsHttpResponseBody:
    body: bytes
    more_body: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiSentUnknown:
    """Any other `type`: a `RuntimeError` for the application."""

    kind: str


AsgiSent: TypeAlias = (
    AsgiSentResponseStart
    | AsgiSentResponseBody
    | AsgiSentResponseTrailers
    | AsgiSentWsAccept
    | AsgiSentWsText
    | AsgiSentWsBytes
    | AsgiSentWsClose
    | AsgiSentWsHttpResponseStart
    | AsgiSentWsHttpResponseBody
    | AsgiSentUnknown
)
"""What the application's `send()` takes."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AsgiFailure:
    """The application raised: `"{Type}: {message}"` and its traceback, for the worker's log."""

    summary: str
    traceback: str
