# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/agent/mod.rs
#     rust/crates/lk-agents-abi/src/agent/assertions.rs
#     rust/crates/lk-agents-abi/src/agent/run.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from livekit.agents._core import SpeechHandle

    from .chat_items import ChatItem, FunctionCall
    from .connect import APIConnectOptions
    from .storage_turn import PendingCall
    from .text_session import FrameId
    from .tool_context import DuplicateMode, DuplicateScope
    from .tool_options import ToolSpec
    from .tool_report import Approval
    from .usage import Usage


@dataclass(kw_only=True, slots=True)
class Invocation:
    """What one host callback runs within, captured when the owning operation was created (a
    `run()` call entry), never from whichever task dispatches it later.
    """

    operation: int
    """This callback, unique within its agent invocation: what [`AgentCallbacks::cancel`] names."""
    invocation_id: str
    """The agent invocation this callback belongs to."""
    agent_name: str
    execution_id: str | None
    frame_id: FrameId | None
    context_scope: str | None
    application_context: object | None
    history: list[ChatItem]
    function_call: FunctionCall | None
    """The tool call, for a tool's preparation, read-back and body."""
    approval: Approval | None
    """The call's approval scope: the body runs on a granted read-back."""
    speech: SpeechHandle | None = None
    """The reply's speech (`_core.SpeechHandle`), for a callback a voice session runs."""


@dataclass(kw_only=True, slots=True)
class ArgumentIdentityJson:
    value: object


@dataclass(kw_only=True, slots=True)
class ArgumentIdentityFingerprint:
    value: str


ArgumentIdentity: TypeAlias = ArgumentIdentityJson | ArgumentIdentityFingerprint
"""The validated arguments a call is identified by: their JSON, or a fingerprint the host
declares for a value JSON cannot represent.
"""


@dataclass(kw_only=True, slots=True)
class PreparedArguments:
    """Arguments the host bound and validated: `prepared` holds them for the body."""

    prepared: object
    identity: ArgumentIdentity


@dataclass(kw_only=True, slots=True)
class ReadBackAnswerQuestion:
    question: str


@dataclass(kw_only=True, slots=True)
class ReadBackAnswerInputRequired:
    """A detail is missing before there is anything to approve."""

    question: str
    verbatim: bool


ReadBackAnswer: TypeAlias = ReadBackAnswerQuestion | ReadBackAnswerInputRequired
"""What a read-back function returned."""


@dataclass(kw_only=True, slots=True)
class OutputCandidateText:
    text: str


@dataclass(kw_only=True, slots=True)
class OutputCandidateValue:
    """`final_result`'s value, unwrapped."""

    value: object


OutputCandidate: TypeAlias = OutputCandidateText | OutputCandidateValue
"""An output candidate for the host to validate."""


@dataclass(kw_only=True, slots=True)
class SerializedOutput:
    """An available JSON representation, including JSON null. Absence means the host value
    has no persistent representation.
    """

    value: object


@dataclass(kw_only=True, slots=True)
class ValidatedOutputAccepted:
    """Accepted: a `str` output's final text, or the validated value the host keeps."""

    text: str | None
    value: object | None
    serialized: SerializedOutput | None = None


@dataclass(kw_only=True, slots=True)
class ValidatedOutputRetry:
    """`ModelRetry` or a validation error: the model corrects its output."""

    reason: str


@dataclass(kw_only=True, slots=True)
class ValidatedOutputFailed:
    """A validator or output function failed outright."""

    error: str


ValidatedOutput: TypeAlias = ValidatedOutputAccepted | ValidatedOutputRetry | ValidatedOutputFailed
"""The host's verdict on an output candidate."""


@dataclass(kw_only=True, slots=True)
class ToolTemplateDuplicateReject:
    """`on_duplicate="reject"` refused a call of `function_name` while `fnc_calls_json` run."""

    function_name: str
    fnc_calls_json: list[str]
    fnc_calls_text: str


@dataclass(kw_only=True, slots=True)
class ToolTemplateDuplicateConfirm:
    """`on_duplicate="confirm"` asks the model to call again to confirm."""

    function_name: str
    fnc_calls_json: list[str]
    fnc_calls_text: str


@dataclass(kw_only=True, slots=True)
class ToolTemplateReplyAtTail:
    """The deferred reply's instructions for the updates `call_ids`, the last still the record's
    tail: the agent has not spoken since.
    """

    call_ids: list[str]


@dataclass(kw_only=True, slots=True)
class ToolTemplateReplyMaybeCovered:
    """The deferred reply's instructions when newer items came after the updates."""

    call_ids: list[str]


ToolTemplate: TypeAlias = (
    ToolTemplateDuplicateReject
    | ToolTemplateDuplicateConfirm
    | ToolTemplateReplyAtTail
    | ToolTemplateReplyMaybeCovered
)
"""A `tool_handling` template the framework asks the host to render (`AsyncToolOptions`), with
the arguments its `*PromptArgs` names.
"""


@dataclass(kw_only=True, slots=True)
class AgentDefinition:
    """The shared agent definition (agents.md "Agent"): reusable configuration every invocation reads
    and none mutates. Host callables (tool bodies, validators, instruction functions, output
    functions, read-backs) stay in the host; the definition says which of them exist.
    """

    name: str
    instructions: str
    """The static instructions."""
    dynamic_instructions: bool
    """`@agent.instructions` functions are registered: the host renders them before each request."""
    tools: list[FunctionToolInfo]
    output: OutputSpec
    retries: Retries


@dataclass(kw_only=True, slots=True)
class FunctionToolInfo:
    """`FunctionToolInfo`: one tool as the model and the gate see it."""

    spec: ToolSpec
    raw_schema: bool = False
    """Declared by a raw JSON schema (`raw_schema=`, MCP): no signature to validate against."""
    needs_approval: ReadBack | None = None
    """`needs_approval=`: the body runs only on an answered read-back."""
    on_duplicate: DuplicateMode = "allow"
    duplicate_scope: DuplicateScope = "name"
    retries: int | None = None
    """`retries=` on the tool: its own budget in place of the run's `Retries.tools`."""
    remote: str | None = None
    """A `RemoteAgent(...).as_tool()`'s endpoint: with the tool's name, its configured identity
    as an expert on a voice session.
    """


ReadBack: TypeAlias = Literal["default", "function"]
"""How a `needs_approval` tool asks."""


@dataclass(kw_only=True, slots=True)
class OutputSpecText:
    """`str`: the model's text is the output. `validated` when output validators are registered."""

    validated: bool


@dataclass(kw_only=True, slots=True)
class OutputSpecStructured:
    """Any other type, or an output function: the parameters' JSON schema. The host validates
    every candidate (type, validators, output function).
    """

    schema: object


OutputSpec: TypeAlias = OutputSpecText | OutputSpecStructured
"""The final output contract (`output_type`)."""


@dataclass(kw_only=True, slots=True)
class Retries:
    """A run's retry budgets (`Agent(retries=)`, `run(retries=)`, pydantic-ai's): how many
    consecutive retries the model gets before the run fails, per tool for argument errors and
    `ModelRetry`, and for the final output.
    """

    tools: int = 1
    output: int = 1


@dataclass(kw_only=True, slots=True)
class EventExpectationMessage:
    """`role` is the `ChatRole` literal."""

    role: str | None


@dataclass(kw_only=True, slots=True)
class EventExpectationFunctionCall:
    """`arguments` holds the keys the call must carry with equal values; others may be present."""

    name: str | None
    arguments: object | None


@dataclass(kw_only=True, slots=True)
class EventExpectationFunctionCallOutput:
    output: str | None
    is_error: bool | None


EventExpectation: TypeAlias = (
    EventExpectationMessage | EventExpectationFunctionCall | EventExpectationFunctionCallOutput
)
"""What an assertion expects of an event: its kind and the fields it names."""


@dataclass(kw_only=True, slots=True)
class RunInputRequirement:
    """One tool call still awaiting human input. This is correlated outcome metadata, not a
    presentation queue: consumers decide when and how to ask, and must retain the call identity.
    """

    invocation_id: str
    call: PendingCall
    question: str
    verbatim: bool


@dataclass(kw_only=True, slots=True)
class RunResult:
    """The settled invocation (`RunResult[Output]`): its status, then the history and usage both
    branches expose.
    """

    outcome: RunOutcome
    interrupted: bool
    """The managed turn yielded at a safe boundary; history and usage still describe its work."""
    new_messages: list[ChatItem]
    """The incoming message and everything this invocation recorded, in order."""
    all_messages: list[ChatItem]
    """The supplied history plus this invocation, without instructions."""
    usage: Usage
    """This invocation's usage, settled (usage.md)."""
    verbatim: str | None
    """The part of the answer or question that reached the person exactly as written."""
    pending_call: PendingCall | None
    """The selected tool call waiting on human input, supplied by the execution core."""
    input_requirements: list[RunInputRequirement] = field(default_factory=list)
    """Every tool-originated input requirement, in model-call order. The singular outcome and
    `pending_call` remain the compatibility projection; other requirements are not answered.
    """


@dataclass(kw_only=True, slots=True)
class RunOutcomeCompleted:
    output: Output


@dataclass(kw_only=True, slots=True)
class RunOutcomeInputRequired:
    """`answer_tool` takes the person's next message as its first string parameter
    (`InputRequired(tool=)`); the next invocation is told so.
    """

    question: str
    verbatim: bool
    answer_tool: str | None


@dataclass(kw_only=True, slots=True)
class RunOutcomeNoOutput:
    """The turn ended with no output: `StopResponse`, a tool result asking for no reply, an
    all-silent step or an interruption. Never `Completed`, whose output is the declared type.
    """


RunOutcome: TypeAlias = RunOutcomeCompleted | RunOutcomeInputRequired | RunOutcomeNoOutput


@dataclass(kw_only=True, slots=True)
class OutputText:
    """A `str` output."""

    text: str


@dataclass(kw_only=True, slots=True)
class OutputValidated:
    """The value the host validated for this candidate; the runtime retains it and the update
    carries it.
    """


Output: TypeAlias = OutputText | OutputValidated
"""The accepted output of a completed invocation."""


@dataclass(kw_only=True, slots=True)
class RunErrorNoModel:
    """The agent has no model to run a step with."""


@dataclass(kw_only=True, slots=True)
class RunErrorCancelled:
    pass


@dataclass(kw_only=True, slots=True)
class RunErrorModel:
    """The model request failed after the provider's own recovery."""

    message: str


@dataclass(kw_only=True, slots=True)
class RunErrorHost:
    """A host callback the invocation depends on (instructions, output validation) failed."""

    message: str


@dataclass(kw_only=True, slots=True)
class RunErrorToolRetriesExceeded:
    """The model kept calling `tool` wrongly past its retry budget."""

    tool: str
    retries: int


@dataclass(kw_only=True, slots=True)
class RunErrorOutputRetriesExceeded:
    """The model kept producing rejected final output past its retry budget."""

    retries: int


RunError: TypeAlias = (
    RunErrorNoModel
    | RunErrorCancelled
    | RunErrorModel
    | RunErrorHost
    | RunErrorToolRetriesExceeded
    | RunErrorOutputRetriesExceeded
)
"""Why an invocation settled without a result."""


@dataclass(kw_only=True, slots=True)
class StreamEventTextDelta:
    message_id: str
    delta: str


@dataclass(kw_only=True, slots=True)
class StreamEventTextEnd:
    message_id: str
    interrupted: bool


@dataclass(kw_only=True, slots=True)
class StreamEventToolCall:
    """A call whose arguments validated, with those arguments."""

    call_id: str
    tool_name: str
    arguments: object


@dataclass(kw_only=True, slots=True)
class StreamEventToolResult:
    call_id: str
    tool_name: str
    status: ToolResultStatus
    output: str


@dataclass(kw_only=True, slots=True)
class StreamEventProgress:
    """`ctx.update(text)`."""

    call_id: str
    tool_name: str
    message: str


StreamEvent: TypeAlias = (
    StreamEventTextDelta
    | StreamEventTextEnd
    | StreamEventToolCall
    | StreamEventToolResult
    | StreamEventProgress
)
"""The incremental view of an invocation (agents.md "Streaming runs"). The coordinator adds the
envelope (sequence, agent name, invocation and frame ids).
"""


ToolResultStatus: TypeAlias = Literal["completed", "failed", "cancelled"]


@dataclass(kw_only=True, slots=True)
class RunRequest:
    """What one invocation is asked: the message, the history it continues, and the host context
    captured at the `run()` entry (a `(loop, contextvars.Context)` pair) its callbacks run in.
    """

    invocation_id: str
    """Prefixes every id the invocation mints."""
    execution_id: str | None = None
    """The transport task id, when this invocation belongs to an A2A task."""
    message: ChatItem
    """A user message."""
    history: list[ChatItem]
    answer_tool: str | None
    """The previous invocation asked with `InputRequired(tool=)`: this message is that tool's."""
    delegated: bool
    """Another agent asked this run: questions pass up as written."""
    conn_options: APIConnectOptions
    host_context: object
    tools_host: object | None = None
    """Canonical host tool objects offered by this invocation."""


@dataclass(kw_only=True, slots=True)
class RunSettled:
    """A settled invocation, with the value the host validated when the output is one."""

    result: RunResult
    output: object | None


@dataclass(kw_only=True, slots=True)
class RunUpdateStream:
    event: StreamEvent


@dataclass(kw_only=True, slots=True)
class RunUpdateIncompleteCleanup:
    """Cancelled work missed its cleanup deadline; the runtime still owns it."""

    pending: list[str]


@dataclass(kw_only=True, slots=True)
class RunUpdateFinished:
    """The last update of a run that settled with a result."""

    settled: RunSettled


@dataclass(kw_only=True, slots=True)
class RunUpdateFailed:
    """The last update of a run that settled without one."""

    error: RunError
    input_requirements: list[RunInputRequirement] = field(default_factory=list)
    provider_error: BaseException | None = None
    """The original model failure, retained beside its correlated outcome metadata."""


RunUpdate: TypeAlias = (
    RunUpdateStream | RunUpdateIncompleteCleanup | RunUpdateFinished | RunUpdateFailed
)
"""What the consumer of a run receives, in order."""
