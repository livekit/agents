# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/voice/session_effects.rs

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from typing_extensions import TypedDict

from .amd import AMDPredictionEvent
from .events import (
    AgentFalseInterruptionEvent,
    AgentStateChangedEvent,
    EotPredictionEvent,
    ToolCallEnded,
    ToolCallUpdated,
    ToolReplyUpdated,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
    _AgentBackchannelOpportunityEvent,
)

if TYPE_CHECKING:
    from livekit.agents._core import SpeechHandle
    from livekit.agents.llm import ChatItem
    from livekit.agents.llm.tool_context import ToolChoice

    from .agent_run import RunError, RunInputRequirement
    from .chat_items import ChatItem, FunctionCall
    from .events import CloseReason
    from .metrics import AgentMetrics
    from .overlap import OverlappingSpeechEvent
    from .rtc_wire import ProcessorFailure
    from .stream_info import StreamInfo
    from .telemetry import RecordingOptions


# Record declarations:
#     rust/crates/lk-agents-abi/src/voice/events.rs
#     rust/crates/lk-agents-abi/src/voice/room_io.rs
#     rust/crates/lk-agents-abi/src/voice/run_view.rs


@dataclass(kw_only=True, slots=True)
class SpeechCreatedWire:
    """`speech_created`'s payload: the host wraps the `_core.SpeechHandle` in its `SpeechHandle` for
    the public `SpeechCreatedEvent` (`voice/events.py`).
    """

    user_initiated: bool
    source: str
    """`SpeechKind::as_source_str`: `"say"` or `"generate_reply"`."""
    speech_handle: SpeechHandle


@dataclass(kw_only=True, slots=True)
class ToolCallStartedWire:
    """A function tool call was dispatched: the call's fields, which the host validates into its
    `llm.FunctionCall`.
    """

    function_call: FunctionCall


@dataclass(kw_only=True, slots=True)
class ToolExecutionUpdatedWire:
    """`tool_execution_updated`'s payload; the host emits `ToolExecutionUpdatedEvent` from it."""

    update: ToolUpdateWire


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
    visibility: str
    """Which host projections take the item: `"both"` (`chat_ctx` and `session.history`) or
    `"history"` (the handoff record).
    """
    run: object | None
    """The open run's host object when an owned speech produced it or it is the handoff
    record, else `None`.
    """


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

    level: str
    """`"debug"`, `"info"` or `"warning"` -- the name `logging` takes."""
    message: str
    fields: dict[str, str]


@dataclass(kw_only=True, slots=True)
class RunItemAddedWire:
    """`run_item_added`'s payload: an item that is the run's before it is committed (a started call)."""

    run: object | None
    """The host's object for the run (its `RunResult`)."""
    item: ChatItem


@dataclass(kw_only=True, slots=True)
class RunStartedWire:
    """`run_started`'s payload."""

    run: object | None
    """The host's object for the run (its run recorder); `None` when the host passed none."""


@dataclass(kw_only=True, slots=True)
class RunRefusedWire:
    """`run_refused`'s payload."""

    run: object | None
    """The host's object for the refused run (its `RunResult`)."""
    cause: str
    """`RunRefusal::as_str`: `"nested"` or `"closing"`."""


@dataclass(kw_only=True, slots=True)
class RunSettledWire:
    """`run_settled`'s payload: the outcome as `RunResult._settle` reads it."""

    input_requirements: list[RunInputRequirement] = field(default_factory=list)
    run: object | None
    """The host's object for the run (its `RunResult`)."""
    outcome: str
    """`RunOutcome::as_str`: `"completed"`, `"input_required"`, `"failed"` or `"closed"`."""
    final_output: object | None
    """The output the run's last reply accepted, resolved; `None` otherwise."""
    question: RunQuestion | None
    """What an `"input_required"` run asks the person."""
    run_error: RunError | None
    """Why the run settled without a result, as `agent.run()` raises it (a tool's retries spent)."""
    speech_handle: SpeechHandle | None
    """The last owned speech's `_core.SpeechHandle`; `None` when the run owned none."""


@dataclass(kw_only=True, slots=True)
class RunQuestion:
    """The question a run ends on (`InputRequired`): asked as written when `verbatim`; the person's
    answer goes to `answer_tool` when one is named.
    """

    question: str
    verbatim: bool
    answer_tool: str | None


@dataclass(kw_only=True, slots=True)
class SpeechDoneWire:
    """`speech_done`'s payload."""

    speech_id: str
    """`speech_{n}`, the handle's id."""
    outcome: str
    """`SpeechOutcome::as_str`: `"completed"`, `"interrupted"` or `"failed"`."""


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
    leg: str | None
    """The session leg whose provider emits the record as its own `metrics_collected`: `"vad"`
    for the VAD's, which the session folds; `None` for one a provider metered itself.
    """


@dataclass(kw_only=True, slots=True)
class OverlappingSpeechWire:
    """`overlapping_speech`'s wire payload: the interruption detector's verdict, as it reported it."""

    event: OverlappingSpeechEvent


@dataclass(kw_only=True, slots=True)
class SessionErrorWire:
    """`error`'s wire payload: the ingredients of `ErrorEvent`; the host wraps `error` in the leg's
    class chosen by `source`.
    """

    source: str
    """`ErrorSource::as_str`."""
    recoverable: bool
    label: str
    """`"host"` when the error was captured off a plugin object, else `"engine"`."""
    error: BaseException
    """The host's own exception when the provider raised one, else an `APIError`."""
    speech_id: str | None
    processor_failure: ProcessorFailure | None
    """An `audio_processing_error`'s processor (`ErrorEvent.source`) and the exception it raised."""


@dataclass(kw_only=True, slots=True)
class HistoryReloadedWire:
    pass


@dataclass(kw_only=True, slots=True)
class ParticipantLinked:
    identity: str


@dataclass(kw_only=True, slots=True)
class ParticipantUnlinked:
    identity: str
    reason: str


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


@dataclass(kw_only=True, slots=True)
class StartPlan:
    """The decision `start()` gets back."""

    recording: RecordingOptions
    is_primary: bool
    hosting: bool
    """`is_primary and (session_host if given else True)`: a second session must never host."""
    records: bool
    """`records_anything` on the resolved options."""


ToolUpdateWire: TypeAlias = ToolCallStartedWire | ToolCallUpdated | ToolCallEnded | ToolReplyUpdated
"""The four tool-lifecycle updates, each crossing as its own record."""


HostWire: TypeAlias = (
    HistoryReloadedWire
    | UserInputTranscribedEvent
    | StartOfSpeechEvent
    | EndOfSpeechEvent
    | SpeechCreatedWire
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
    | RunItemAddedWire
    | RunRefusedWire
    | RunSettledWire
    | SpeechDoneWire
    | FloorChangedWire
    | CloseWire
    | MetricsCollectedWire
    | OverlappingSpeechWire
    | SessionErrorWire
    | ToolExecutionUpdatedWire
    | ParticipantLinked
    | ParticipantUnlinked
    | TextInputWire
    | DeleteRoomRequested
)
"""Every event the session hands its host, as the record the host reads, routed by its class.

`#[host(event)]` members are emitted as public session events and `#[host(ignored)]` ones are
dropped, both via
`_generated/session_dispatch.py`; the rest have host behaviour.
"""


VoiceRunOutput: TypeAlias = str | object


DEFAULT_HOOK_BOUND: Final = 15.0
DEFAULT_MAX_UNRECOVERABLE_ERRORS: Final = 3
DEFAULT_SPEAKING_FPS: Final = 1.0
DEFAULT_SILENT_FPS: Final = 0.3
