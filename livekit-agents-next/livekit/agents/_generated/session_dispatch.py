# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any

from .events import (
    AgentFalseInterruptionEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
)


# `HostWire` members the host emits as a public session event and nothing more:
# {wire kind: (event class, event name)}. The payload's keys are the class's fields.
PUBLIC_EVENTS: dict[str, tuple[type[Any], str]] = {
    "user_turn_exceeded": (UserTurnExceededEvent, "user_turn_exceeded"),
    "transcription_timeout": (UserTranscriptionTimeoutEvent, "user_transcription_timeout"),
    "agent_false_interruption": (AgentFalseInterruptionEvent, "agent_false_interruption"),
}

# `HostWire` members the host receives and does not act on.
IGNORED_EVENTS: frozenset[str] = frozenset(
    {
        "user_turn_completed",
        "agent_backchannel_opportunity",
        "preemptive_generation",
        "backchannel_confirmed",
        "overlap_interruption",
    }
)

# Every kind the stream carries: each `HostWire` member's, and the records it carries beside them.
EVENT_KINDS: frozenset[str] = frozenset(
    {
        "user_input_transcribed",
        "user_started_speaking",
        "user_stopped_speaking",
        "speech_created",
        "diagnostic",
        "function_tools_executed",
        "user_turn_completed",
        "eot_prediction",
        "agent_backchannel_opportunity",
        "preemptive_generation",
        "user_turn_exceeded",
        "transcription_timeout",
        "backchannel_confirmed",
        "overlap_interruption",
        "span_started",
        "span_ended",
        "agent_state_changed",
        "user_state_changed",
        "agent_false_interruption",
        "amd_result",
        "conversation_item_added",
        "context_carried",
        "run_started",
        "run_item_added",
        "run_refused",
        "run_settled",
        "speech_done",
        "floor_changed",
        "close",
        "metrics_collected",
        "error",
        "tool_execution_updated",
        "hook_call",
        "participant_linked",
        "participant_unlinked",
        "text_input",
        "delete_room_requested",
        "stream_closed",
    }
)
