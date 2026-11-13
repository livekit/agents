# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/utils/socket.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import get_annotations
from typing import TypeAlias


@dataclass
class SocketTarget:
    """Where the framework opens a provider socket: through the job's HTTP proxy, pinging so a
    half-open socket is noticed. A rejected handshake's request id is read from
    `request_id_header`.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    protocols: list[str] = field(default_factory=list)
    request_id_header: str | None = None


@dataclass(kw_only=True, slots=True)
class KeepaliveFrame:
    """A frame sent once nothing else was sent for `interval` seconds."""

    interval: float
    frame: WsFrame


@dataclass(kw_only=True, slots=True)
class SocketConnection:
    """One connection as the protocol asks for it, read at each connection: options changed since
    the last one apply to the next.
    """

    target: SocketTarget
    keepalive: KeepaliveFrame | None
    goodbye: WsFrame | None
    """Sent once the input ended and before a rotation; the provider then drains and closes."""
    max_session_duration: float | None
    """Seconds after which the connection is rotated, at the next input boundary."""
    proxy: str | None
    """The job's HTTP proxy, as the host's HTTP session uses it."""


@dataclass(kw_only=True, slots=True)
class SocketSend:
    """The frames of the next input item, encoded. `boundary`: it ended a segment (a flush), where a
    due rotation may run. `end`: it was the last item.
    """

    frames: list[WsFrame]
    boundary: bool
    end: bool


WsFrame: TypeAlias = str | bytes
"""One websocket frame: text, or a binary payload."""


SocketTarget.__annotations__ = get_annotations(SocketTarget, eval_str=True)
SocketTarget.__module__ = "livekit.agents.http"
