# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/voice/session_effects.rs

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import ConfigDict
from typing_extensions import TypedDict

from .amd import AMDPredictionEvent
from .events import (
    AgentFalseInterruptionEvent,
    AgentStateChangedEvent,
    EotPredictionEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
    _AgentBackchannelOpportunityEvent,
)
from .overlap import OverlappingSpeechEvent

if TYPE_CHECKING:
    from typing import Annotated

    from pydantic import Field

    from livekit.agents.llm import ChatItem
    from livekit.agents.llm.tool_context import ToolChoice
    from livekit.agents.voice.io import SpeechHandle as HostSpeechHandle

    from .agent_run import RunFinished, RunInputRequirement
    from .events import CloseReason
    from .metrics import AgentMetrics
    from .rtc_wire import ProcessorFailure
    from .session_events import ToolUpdate
    from .stream_info import StreamInfo


# Record declarations:
#     rust/crates/lk-agents-abi/src/voice/events.rs
#     rust/crates/lk-agents-abi/src/voice/room_io.rs


@dataclass(kw_only=True, slots=True)
class ToolExecutionUpdatedWire:
    """`tool_execution_updated`'s payload; the host emits `ToolExecutionUpdatedEvent` from it."""

    update: ToolUpdate


@dataclass(kw_only=True, slots=True)
class StartOfSpeechEvent:
    """The user started speaking (`on_start_of_speech`): the record is the whole message."""


@dataclass(kw_only=True, slots=True)
class PreemptiveGenerationEvent:
    """A transcript stable enough to start generating against (`_PreemptiveGenerationInfo`): the record
    is the whole message.
    """


@dataclass(kw_only=True, slots=True)
class BackchannelConfirmedWire:
    """A backchannel was confirmed: the record is the whole message."""


@dataclass(kw_only=True, slots=True)
class OverlapInterruptionEvent:
    """The overlap was a real interruption (`on_interruption`): the record is the whole message."""


@dataclass(kw_only=True, slots=True)
class UserTurnCompletedWire:
    """`user_turn_completed`'s payload."""

    turn: int
    """The turn's id: what `hook_done(turn, ..)` answers for."""
    new_transcript: str
    transcript_confidence: float


@dataclass(kw_only=True, slots=True)
class EndOfSpeechEvent:
    """The user stopped speaking (`on_end_of_speech`)."""

    silence_duration: float
    """Silence already elapsed when the event was raised, in seconds (`0.0` on the STT path). `at`
    includes it, but a host needs it: AMD's post-speech window runs from when speech ended.
    """


@dataclass(kw_only=True, slots=True)
class ItemAddedWire:
    """`conversation_item_added`'s payload."""

    host_item: ChatItem | None
    """The item as the session's history holds it: the host's `llm.ChatItem`, the same object
    `session.history` returns. `None` only for a host that made no object for it.
    """
    agent: str
    """Which agent's frame produced it, as the opaque id the host gave the core."""
    visibility: Literal["model", "history", "both"]


@dataclass(kw_only=True, slots=True)
class FunctionToolsExecutedWire:
    """`function_tools_executed`, as the driver emits it; calls and outputs are parallel lists. The
    host ignores it: its `FunctionToolsExecutedEvent` comes from the `function_tools_executed`
    hook, and the remote session forwards these lists.
    """


@dataclass(kw_only=True, slots=True)
class DiagnosticWire:
    """A sentence the core decided and the host logs. `fields` is the reference's `extra={...}`, so
    `lk.pii.*` entries go through the host's redaction filter.
    """

    level: Literal["info", "warning", "error"]
    message: str
    fields: dict[str, str]


@dataclass(kw_only=True, slots=True)
class RunStartedWire:
    """`run_started`'s payload."""

    run: object | None
    """The host's object for the run (the future its `_run()` awaits); `None` when the host
    passed none.
    """


@dataclass(kw_only=True, slots=True)
class RunRefusedWire:
    """`run_refused`'s payload."""

    run: object | None
    """The host's object for the refused run (the future its `_run()` awaits)."""
    cause: Literal["nested", "closing"]


@dataclass(kw_only=True, slots=True)
class RunFinishedWire:
    """`run_finished`'s payload: the run settled with a result, as a text run's `Finished` update."""

    run: object | None
    """The host's object for the run (the future its `_run()` awaits)."""
    settled: RunFinished


@dataclass(kw_only=True, slots=True)
class RunFailedWire:
    """`run_failed`'s payload: the run settled without a result, as a text run's `Failed` update."""

    run: object | None
    """The host's object for the run (the future its `_run()` awaits)."""
    error: BaseException
    input_requirements: list[RunInputRequirement] = field(default_factory=list)
    provider_error: BaseException | None = None
    """The provider failure the run's last speech ended on, raised as itself."""


@dataclass(kw_only=True, slots=True)
class SpeechDoneWire:
    """`speech_done`'s payload."""

    speech_id: str
    """`speech_{n}`, the handle's id."""
    outcome: Literal["completed", "interrupted", "failed"]


@dataclass(kw_only=True, slots=True)
class FloorChangedWire:
    """`floor_changed`'s payload. `speech_id` is `None` when the floor is free."""

    speech_id: str | None


@dataclass(kw_only=True, slots=True)
class ContextCarriedWire:
    """`context_carried`'s payload: item ids moved from `from_agent`'s context to `to_agent`'s."""

    from_agent: str
    to_agent: str
    items: list[str]


@dataclass(kw_only=True, slots=True)
class CloseWire:
    """`close`'s wire payload: `error` is the unrecoverable `error` event that closed the session, the
    key absent when none did.
    """

    reason: CloseReason
    error: SessionErrorWire | None


@dataclass(kw_only=True, slots=True)
class MetricsCollectedWire:
    """`metrics_collected`'s wire payload: the record plus the driver's `speech_id` correlation."""

    metrics: AgentMetrics
    """The `AgentMetrics` record as the dict `TypeAdapter(AgentMetrics)` validates."""
    speech_id: str | None
    """`speech_{n}` of the speech the record belongs to; `None` for a leg that has none."""
    leg: MetricsLeg | None
    """The session leg whose provider emits the record as its own `metrics_collected`: `"vad"`
    for the VAD's, which the session folds; `None` for one a provider metered itself.
    """


MetricsLeg: TypeAlias = Literal["vad"]
"""[`MetricsCollectedWire::leg`]: the session leg whose provider a record belongs to."""


@dataclass(kw_only=True, slots=True)
class SessionErrorWire:
    """`error`'s wire payload: the ingredients of `ErrorEvent`; the host wraps `error` in the leg's
    class chosen by `source`.
    """

    source: Literal["stt_error", "llm_error", "tts_error", "realtime_model_error", "audio_processing_error"]
    slot: Literal["llm", "stt", "tts", "vad"] | None
    """The session slot whose provider failed (`source.slot()`); `None` for an audio processor."""
    recoverable: bool
    label: Literal["host", "engine"]
    error: BaseException
    """The host's own exception when the provider raised one, else an `APIError`."""
    speech_id: str | None
    processor_failure: ProcessorFailure | None
    """An `audio_processing_error`'s processor (`ErrorEvent.source`) and the exception it raised."""


@dataclass(kw_only=True, slots=True)
class HistoryReloadedWire:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class SpeechCreatedEvent:
    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    user_initiated: bool
    """True if the speech was created using public methods like `say` or `generate_reply`"""
    source: Literal["say", "generate_reply"]
    """Source indicating how the speech handle was created"""
    speech_handle: Annotated[HostSpeechHandle, Field(exclude=True)] = field(repr=False)
    """The speech handle that was created"""
    created_at: float = field(default_factory=time.time)
    type: Literal["speech_created"] = "speech_created"


@dataclass(kw_only=True, slots=True)
class ParticipantLinked:
    identity: str


@dataclass(kw_only=True, slots=True)
class ParticipantUnlinked:
    identity: str
    reason: Literal["unknown", "client_initiated", "duplicate_identity", "server_shutdown", "participant_removed", "room_deleted", "state_mismatch", "join_failure", "migration", "signal_close", "room_closed", "user_unavailable", "user_rejected", "sip_trunk_failure", "connection_timeout", "media_failure", "agent_error"]


@dataclass(kw_only=True, slots=True)
class TextInputWire:
    """`text_input`'s payload: the host builds its `TextInputEvent` from it, with the participant and
    `info` as its SDK objects.
    """

    text: str
    participant: str
    info: StreamInfo


@dataclass(kw_only=True, slots=True)
class DeleteRoomRequested:
    room: str


@dataclass(kw_only=True, slots=True)
class DefaultModelPlan:
    vad_model: DefaultVadModel | None
    turn_detector: bool


DefaultVadModel: TypeAlias = Literal["silero"]


class ExceededTurnReply(TypedDict):
    instructions: str
    allow_interruptions: bool
    tool_choice: ToolChoice


HostWire: TypeAlias = (
    HistoryReloadedWire
    | UserInputTranscribedEvent
    | StartOfSpeechEvent
    | EndOfSpeechEvent
    | SpeechCreatedEvent
    | DiagnosticWire
    | FunctionToolsExecutedWire
    | UserTurnCompletedWire
    | EotPredictionEvent
    | _AgentBackchannelOpportunityEvent
    | PreemptiveGenerationEvent
    | UserTurnExceededEvent
    | UserTranscriptionTimeoutEvent
    | BackchannelConfirmedWire
    | OverlapInterruptionEvent
    | AgentStateChangedEvent
    | UserStateChangedEvent
    | AgentFalseInterruptionEvent
    | AMDPredictionEvent
    | ItemAddedWire
    | ContextCarriedWire
    | RunStartedWire
    | RunRefusedWire
    | RunFinishedWire
    | RunFailedWire
    | SpeechDoneWire
    | FloorChangedWire
    | CloseWire
    | MetricsCollectedWire
    | OverlappingSpeechEvent
    | SessionErrorWire
    | ToolExecutionUpdatedWire
    | ParticipantLinked
    | ParticipantUnlinked
    | TextInputWire
    | DeleteRoomRequested
)
"""Every event the session hands its host, as the record the host reads, routed by its class.

`#[host(event)]` members are emitted as public session events and `#[host(ignored)]` ones are
dropped, both via `_generated/session_dispatch.py`; the rest have host behaviour.
`#[host(emits = "Model")]` names the public event that behaviour emits, so the generated
`EventTypes` and `AgentEvent` list every public session event.
"""


SpeechCreatedEvent.__module__ = "livekit.agents.voice.events"


DEFAULT_HOOK_BOUND: Final = 15.0
DEFAULT_MAX_UNRECOVERABLE_ERRORS: Final = 3
