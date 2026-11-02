# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/worker/http/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass(frozen=True)
class _HttpServerLimits:
    """Resource limits of the host HTTP/ASGI adapter."""

    request_head_bytes: int = 1048576
    body_metadata_bytes: int = 65536
    close_flush_timeout: float = 1.0
    websocket_frame_bytes: int = 1048576
    websocket_message_bytes: int = 4194304
    websocket_inbox: int = 16
    websocket_inbox_bytes: int = 8388608
    websocket_outbox_bytes: int = 8388608
    pending_handlers: int = 256
    payload_bytes: int = 536870912


_HttpServerLimits.__annotations__ = get_annotations(_HttpServerLimits, eval_str=True)
_HttpServerLimits.__module__ = "livekit.agents.http"
