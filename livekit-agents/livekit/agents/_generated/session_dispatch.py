# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/events.rs

from typing import Annotated, Literal

from pydantic import Field

from .events import (
    AgentFalseInterruptionEvent,
    AgentStateChangedEvent,
    OverlappingSpeechEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnEndedEvent,
    UserTurnExceededEvent,
    UserTurnStartedEvent,
    _AgentBackchannelOpportunityEvent,
)
from .session_events import (
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    ToolExecutionUpdatedEvent,
)
from .session_wire import (
    BackchannelConfirmedWire,
    FunctionToolsExecutedWire,
    HistoryReloadedWire,
    OverlapInterruptionEvent,
    ParticipantLinked,
    ParticipantUnlinked,
    PreemptiveGenerationEvent,
    SpeechCreatedEvent,
    SpeechDoneWire,
    UserTurnCompletedWire,
)

# `HostWire` members the host emits as public session events: {model class: event name}.
PUBLIC_EVENTS: dict[type, str] = {
    UserInputTranscribedEvent: "user_input_transcribed",
    SpeechCreatedEvent: "speech_created",
    UserTurnExceededEvent: "user_turn_exceeded",
    UserTurnStartedEvent: "user_turn_started",
    UserTurnEndedEvent: "user_turn_ended",
    UserTranscriptionTimeoutEvent: "user_transcription_timeout",
    AgentStateChangedEvent: "agent_state_changed",
    UserStateChangedEvent: "user_state_changed",
    AgentFalseInterruptionEvent: "agent_false_interruption",
    OverlappingSpeechEvent: "overlapping_speech",
}

# `HostWire` members the host receives and does not act on.
IGNORED_EVENTS: frozenset[type] = frozenset(
    {
        HistoryReloadedWire,
        FunctionToolsExecutedWire,
        UserTurnCompletedWire,
        _AgentBackchannelOpportunityEvent,
        PreemptiveGenerationEvent,
        BackchannelConfirmedWire,
        OverlapInterruptionEvent,
        SpeechDoneWire,
        ParticipantLinked,
        ParticipantUnlinked,
    }
)

# Every public session event's name: the routed members' and those the host's behaviour emits.
EventTypes = Literal[
    "user_input_transcribed",
    "speech_created",
    "function_tools_executed",
    "user_turn_exceeded",
    "user_turn_started",
    "user_turn_ended",
    "user_transcription_timeout",
    "agent_state_changed",
    "user_state_changed",
    "agent_false_interruption",
    "conversation_item_added",
    "close",
    "overlapping_speech",
    "error",
    "tool_execution_updated",
]

# Every public session event, told apart by its `type`.
AgentEvent = Annotated[
    UserInputTranscribedEvent
    | SpeechCreatedEvent
    | FunctionToolsExecutedEvent
    | UserTurnExceededEvent
    | UserTurnStartedEvent
    | UserTurnEndedEvent
    | UserTranscriptionTimeoutEvent
    | AgentStateChangedEvent
    | UserStateChangedEvent
    | AgentFalseInterruptionEvent
    | ConversationItemAddedEvent
    | CloseEvent
    | OverlappingSpeechEvent
    | ErrorEvent
    | ToolExecutionUpdatedEvent,
    Field(discriminator="type"),
]
