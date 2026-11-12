# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

def flush_logs() -> None:
    """Waits until every line the core logged so far has reached the host's logging. Tests call
    it at their boundaries; the interpreter's exit calls it.
    """
