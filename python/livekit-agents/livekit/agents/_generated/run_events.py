# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/agent/stream.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import Literal

from .chat_items import ChatMessage, FunctionCall, FunctionCallOutput


@dataclass
class ChatMessageEvent:
    item: ChatMessage
    type: Literal["message"] = "message"


@dataclass
class FunctionCallEvent:
    item: FunctionCall
    type: Literal["function_call"] = "function_call"


@dataclass
class FunctionCallOutputEvent:
    item: FunctionCallOutput
    type: Literal["function_call_output"] = "function_call_output"


ChatMessageEvent.__annotations__ = get_annotations(ChatMessageEvent, eval_str=True)
ChatMessageEvent.__module__ = "livekit.agents.voice.run_result"
FunctionCallEvent.__annotations__ = get_annotations(FunctionCallEvent, eval_str=True)
FunctionCallEvent.__module__ = "livekit.agents.voice.run_result"
FunctionCallOutputEvent.__annotations__ = get_annotations(FunctionCallOutputEvent, eval_str=True)
FunctionCallOutputEvent.__module__ = "livekit.agents.voice.run_result"
