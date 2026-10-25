# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/llm/realtime.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RealtimeCapabilities:
    message_truncation: bool
    """Whether generated assistant messages can be truncated after interruption"""
    turn_detection: bool
    """Whether the model emits server-side speech start and stop events for turn taking"""
    user_transcription: bool
    """Whether the model emits user audio transcription events"""
    auto_tool_reply_generation: bool
    """Whether the model automatically generates a reply after receiving tool results"""
    audio_output: bool
    """Whether the model can produce audio output directly"""
    manual_function_calls: bool
    """Whether function call items already in the chat context can be resumed"""
    can_disable_turn_detection: bool = False
    """Whether server-side turn detection can be disabled so the client drives turn-taking."""
    mutable_chat_context: bool = False
    """Whether the chat context can be updated mid-session"""
    mutable_instructions: bool = False
    """Whether the instructions can be updated mid-session"""
    mutable_tools: bool = False
    """Whether the tools can be updated mid-session"""
    per_response_tool_choice: bool = False
    """Whether the tool and tool choice can be specified per response"""
    supports_say: bool = False
    """Whether session.say() can use the realtime session directly, without TTS.

    When used through a RealtimeModel, add_to_chat_ctx=False is ignored and the
    message is still added to the chat context.
    """
    auto_turn_reply_generation: bool = False
    """Whether the server generates the reply by itself when its turn detection ends a user turn.
    Last: the reference has no such field, and positional arguments must keep its order.
    """


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
RealtimeCapabilities.__module__ = "livekit.agents.llm.realtime"
