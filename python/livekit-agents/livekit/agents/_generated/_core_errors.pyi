# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from .types import ErrorKind

def http_error_display(message: str, status: int, reason: str | None) -> str:
    """The message of an `APIStatusError` built from an HTTP status: the message and the reason
    when the message adds detail, else just the reason.
    """

def error_kind_for_status(status_code: int, kind: ErrorKind | None) -> ErrorKind:
    """`APIStatusError`'s kind: the one its caller named, else its status's."""

def error_kind_retryable(kind: ErrorKind) -> bool:
    """`APIError.retryable`: whether its kind is worth another attempt."""

def websocket_close_kind(code: int | None) -> ErrorKind | None:
    """What a close code says failed: `http.WebSocket.close_error`'s kind, `None` for a close on
    purpose.
    """
