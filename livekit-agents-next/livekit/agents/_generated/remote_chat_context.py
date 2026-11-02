# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/chat_context/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations

from livekit.agents.llm.chat_context import ChatItem


@dataclass
class _RemoteChatItem:
    """A host item view keeps the Python ChatItem identity; native code owns only its order."""

    item: ChatItem


_RemoteChatItem.__annotations__ = get_annotations(_RemoteChatItem, eval_str=True)
_RemoteChatItem.__module__ = "livekit.agents.llm.remote_chat_context"
