# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/utils/websocket.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import get_annotations
from typing import Literal, TypeAlias


@dataclass
class WebSocketTarget:
    """Where the framework opens a provider WebSocket: through the job's HTTP proxy, pinging so a
    half-open socket is noticed. A rejected handshake's request id is read from
    `request_id_header`.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    protocols: list[str] = field(default_factory=list)
    request_id_header: str | None = None


@dataclass(kw_only=True, slots=True)
class WebSocketKeepaliveFrame:
    """A frame sent once nothing else was sent for `interval` seconds."""

    interval: float
    frame: WsFrame


@dataclass(kw_only=True, slots=True)
class WebSocketConnection:
    """One connection as the protocol asks for it, read at each connection: options changed since
    the last one apply to the next.
    """

    target: WebSocketTarget
    keepalive: WebSocketKeepaliveFrame | None
    max_session_duration: float | None
    """Seconds after which the connection is replaced, at the stream's first moment it can be."""
    ping_interval: float | None
    """Seconds between pings; a ping unanswered for half of it fails the connection. `None`
    sends none.
    """
    setup: list[WsFrame]
    """Sent first on the connection, before any input or control message: the provider
    session's setup.
    """
    awaits_ready: bool
    """The provider must signal `ready` before anything but the setup is sent; it has the
    connect limit to say so, else the attempt fails with a timeout.
    """
    input_idle: float | None
    """Seconds without input after which the stream's input-idle frames go, once per silence."""
    context_keepalive: float | None
    """A shared socket's: seconds without a send on a context after which the stream's open
    contexts are kept alive.
    """


@dataclass(kw_only=True, slots=True)
class WebSocketInput:
    """One input item's frames, encoded, and the seconds they play for, which real-time pacing
    spaces them by.
    """

    frames: list[WsFrame]
    pace: float


@dataclass(kw_only=True, slots=True)
class WebSocketRead:
    """A received message, handled: what it tells the connection, and the provider's id for the
    request or response it belongs to, when it names one.
    """

    signal: Literal["ready", "session_finished"] | None
    request_id: str | None


@dataclass(kw_only=True, slots=True)
class WebSocketMuxRoute:
    """Where a shared socket's received frame goes: to `context` as `message` (decoded once, by the
    multiplexer), or to the socket itself (`None`); `ready`: the socket's provider session is ready
    for streams.
    """

    context: str | None
    message: object
    ready: bool


@dataclass(kw_only=True, slots=True)
class WebSocketMuxOptions:
    """How a multiplexer shares its sockets. `max_contexts`: streams one socket carries at once.
    `max_sockets`: sockets open at once; a stream finding none with room waits for one within
    the pool limit. `idle_timeout`: seconds a socket with no stream stays open. `keep_warm`: an
    idle socket the provider closed is opened again. `routed`: each frame is routed to its
    context; an unrouted socket carries one stream, gets every frame, and is reused only after a
    stream that ended cleanly.
    """

    max_contexts: int | None
    max_sockets: int | None
    idle_timeout: float | None
    keep_warm: bool
    routed: bool


WsFrame: TypeAlias = str | bytes
"""One websocket frame: text, or a binary payload."""


WebSocketTarget.__annotations__ = get_annotations(WebSocketTarget, eval_str=True)
WebSocketTarget.__module__ = "livekit.agents._websocket"
