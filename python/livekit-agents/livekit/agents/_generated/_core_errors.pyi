# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

def http_error_display(message: str, status: int, reason: str | None) -> str:
    """The message of an `APIStatusError` built from an HTTP status: the message and the reason
    when the message adds detail, else just the reason.
    """

def retryable_for_status(status_code: int, retryable: bool | None) -> bool:
    """`APIStatusError`'s `retryable`: a non-transient 4xx status, 499 among them, never is."""
