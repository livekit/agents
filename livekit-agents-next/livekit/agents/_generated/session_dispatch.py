# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/events.rs

from .events import (
    AgentFalseInterruptionEvent,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
    _AgentBackchannelOpportunityEvent,
)
from .session_wire import (
    BackchannelConfirmedWire,
    OverlapInterruptionEvent,
    ParticipantLinked,
    ParticipantUnlinked,
    PreemptiveGenerationEvent,
    UserTurnCompletedWire,
)

# `HostWire` members the host emits as public session events: {model class: event name}.
PUBLIC_EVENTS: dict[type, str] = {
    UserInputTranscribedEvent: "user_input_transcribed",
    UserTurnExceededEvent: "user_turn_exceeded",
    UserTranscriptionTimeoutEvent: "user_transcription_timeout",
    AgentStateChangedEvent: "agent_state_changed",
    UserStateChangedEvent: "user_state_changed",
    AgentFalseInterruptionEvent: "agent_false_interruption",
}

# Public events whose `new_state` the host keeps on the session: {model class: attribute}.
MIRRORED_STATES: dict[type, str] = {
    AgentStateChangedEvent: "_agent_state",
    UserStateChangedEvent: "_user_state",
}

# `HostWire` members the host receives and does not act on.
IGNORED_EVENTS: frozenset[type] = frozenset(
    {
        UserTurnCompletedWire,
        _AgentBackchannelOpportunityEvent,
        PreemptiveGenerationEvent,
        BackchannelConfirmedWire,
        OverlapInterruptionEvent,
        ParticipantLinked,
        ParticipantUnlinked,
    }
)
