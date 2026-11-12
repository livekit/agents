# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/session_effects.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass

from .chat_items import ChatItem


@dataclass
class CompletedUserTurn:
    """What `on_user_turn_completed` left: the message's text as the hook left it, the turn's items
    when it changed them, and whether it raised `StopResponse`.
    """

    user_text: str | None
    items: list[ChatItem] | None
    stop_response: bool
