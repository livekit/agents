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
    `request_id_header`. `max_message_size`: the largest message (and frame) the provider may
    send, in bytes; `None` keeps the transport's limit.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    protocols: list[str] = field(default_factory=list)
    request_id_header: str | None = None
    max_message_size: int | None = None


@dataclass(kw_only=True, slots=True)
class WebSocketKeepaliveFrame:
    """A frame sent once nothing else was sent for `interval` seconds. `only_while_unused`: on a
    multiplexer's socket, sent only while no stream holds it.
    """

    interval: float
    frame: WsFrame
    only_while_unused: bool


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
    goodbye: list[WsFrame]
    """The frames that ask the provider to end its session, sent when a replacement is due
    (`max_session_duration`) and when the input ends: the connection then takes no more input
    and ends, or is replaced, once the provider finished its session or closed cleanly, or the
    read limit passed. Empty: it ends or is replaced at once. A shared socket's own goodbye is
    its multiplexer's.
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
    response: bool
    """Response data arrived; control frames and unknown messages are not progress."""


@dataclass(kw_only=True, slots=True)
class WebSocketMuxRoute:
    """Where a shared socket's received frame goes: to `context` as `message` (decoded once, by the
    multiplexer), or to the socket itself (`None`); `ready`: the socket's provider session is ready
    for streams; `ended_context`: the context the message ends, the provider's last word on it
    (what a cancel waits for).
    """

    context: str | None
    message: object
    ready: bool
    ended_context: str | None


@dataclass(kw_only=True, slots=True)
class WebSocketContextsCancel:
    """The frames that cancel the contexts a stream left before the provider finished them, and
    those contexts.
    """

    contexts: list[str]
    frames: list[WsFrame]


@dataclass(kw_only=True, slots=True)
class WebSocketMuxOptions:
    """How a multiplexer shares its sockets. `max_contexts`: streams one socket carries at once.
    `max_sockets`: sockets open at once; a stream finding none with room waits for one within
    the pool limit. `idle_timeout`: seconds a socket with no stream stays open. `keep_warm`: a
    socket with room is kept open as a standby: one opens when a stream takes the last, and an
    idle one the provider closed is opened again. `routed`: each frame is routed to its context;
    an unrouted socket carries one stream, gets every frame, and is reused only after a stream
    that ended cleanly. `reuses_sockets`: `false` closes a socket once its stream left it, so
    each stream (each session, with a connection per segment) has a fresh one.
    `reads_cancel_end`: an unrouted socket a stream left mid-answer is cancelled with the stream's
    frames and reused once the route reads the end of every context cancelled, within
    [`CANCELLED_ANSWER_WAIT`]; otherwise it is closed.
    """

    max_contexts: int | None
    max_sockets: int | None
    idle_timeout: float | None
    keep_warm: bool
    routed: bool
    reuses_sockets: bool
    reads_cancel_end: bool


WsFrame: TypeAlias = str | bytes
"""One websocket frame: text, or a binary payload."""


WebSocketTarget.__annotations__ = get_annotations(WebSocketTarget, eval_str=True)
WebSocketTarget.__module__ = "livekit.agents._websocket"
