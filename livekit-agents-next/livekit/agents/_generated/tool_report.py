# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/agent/mod.rs
#     rust/crates/lk-agents-abi/src/llm/tool_executor.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import JsonValue

from livekit.agents._generated.function_tools import InputRequired
from livekit.agents._generated.tool_errors import ModelRetry, StopResponse, ToolError
from livekit.agents._generated.tool_outcomes import Failed, Rejected, ToolResult, Working
from livekit.agents.llm.tool_context import Completed

from .usage import Usage


@dataclass
class ToolCompletion:
    """What a tool's run reports: its outcome, the call's `extra` as the body left it, and what a
    delegate it ran in process (an agent tool) consumed, which the caller's usage includes.
    """

    report: ToolReport
    extra: dict[str, JsonValue]
    usage: Usage | None = None


@dataclass(kw_only=True, slots=True)
class ToolOutcomeValue:
    """A value the model reads: `text` is a string as written, anything else as its JSON.
    `ToolResult(..)` sets `reply_required` and `verbatim`; `more` is what a delegate's
    verbatim result carries for the model alone.
    """

    text: str
    reply_required: bool = True
    verbatim: bool = False
    more: str | None = None


@dataclass(kw_only=True, slots=True)
class ToolOutcomeNothing:
    """`None`: nothing to say, no reply asked."""


@dataclass(kw_only=True, slots=True)
class ToolOutcomeToolError:
    """`ToolError(message)`: the author's words, for the model."""

    message: str


@dataclass(kw_only=True, slots=True)
class ToolOutcomeModelRetry:
    """`ModelRetry(message)`: the reason, for the model to correct its call."""

    message: str


@dataclass(kw_only=True, slots=True)
class ToolOutcomeStopResponse:
    """`StopResponse`: answered with nothing, no reply."""


@dataclass(kw_only=True, slots=True)
class ToolOutcomeInternal:
    """Any other exception; the host logged it, the model is told nothing about it."""


@dataclass(kw_only=True, slots=True)
class ToolOutcomeInvalid:
    """A value the model can't be told about (`_is_valid_function_output`)."""


@dataclass(kw_only=True, slots=True)
class ToolOutcomeCancelled:
    """The host cancelled the body."""


@dataclass(kw_only=True, slots=True)
class ToolOutcomeUpdate:
    """`ctx.update(message)`: `message` rendered, `raw` as written; the tool runs on."""

    message: str
    raw: str


@dataclass(kw_only=True, slots=True)
class ToolOutcomeCompleted:
    """`Completed(value)`: the value as text (`completed` for `None`)."""

    text: str
    verbatim: bool = False


@dataclass(kw_only=True, slots=True)
class ToolOutcomeFailed:
    error: str


@dataclass(kw_only=True, slots=True)
class ToolOutcomeRejected:
    reason: str


@dataclass(kw_only=True, slots=True)
class ToolOutcomeInputRequired:
    """`InputRequired(question)`: the turn waits on the person; `tool` takes the answer."""

    question: str
    verbatim: bool = False
    tool: str | None = None


@dataclass(kw_only=True, slots=True)
class ToolOutcomeWorking:
    """`Working(message=)`: waits on a system, nothing delivered."""

    message: str | None = None


ToolOutcome: TypeAlias = (
    ToolOutcomeValue
    | ToolOutcomeNothing
    | ToolOutcomeToolError
    | ToolOutcomeModelRetry
    | ToolOutcomeStopResponse
    | ToolOutcomeInternal
    | ToolOutcomeInvalid
    | ToolOutcomeCancelled
    | ToolOutcomeUpdate
    | ToolOutcomeCompleted
    | ToolOutcomeFailed
    | ToolOutcomeRejected
    | ToolOutcomeInputRequired
    | ToolOutcomeWorking
)
"""What a tool's run came to, as the host classified the value it returned or raised. The host
only classifies; the outcome table in `lk-agents-core` decides what it means.
"""


Approval: TypeAlias = Literal["asked", "granted"]
"""How a call of a tool that requires approval went: it asked for approval with its read-back,
or ran on the person's answer. Marked on the call for the next to read.
"""


@dataclass(kw_only=True, slots=True)
class CompletedReturn:
    """`Completed(value, verbatim=)`."""

    value: object
    verbatim: bool


@dataclass(kw_only=True, slots=True)
class InputRequiredReturn:
    """`InputRequired(question, tool=, verbatim=)`: `tool` the name of the tool taking the answer."""

    question: str
    verbatim: bool
    tool: str | None


@dataclass(kw_only=True, slots=True)
class ToolResultReturn:
    """`ToolResult(output, reply_required=, verbatim=)`; `_more` is what a delegate's verbatim
    result carries for the model alone.
    """

    output: object
    reply_required: bool
    verbatim: bool
    _more: str | None


@dataclass(kw_only=True, slots=True)
class ToolReport:
    """The host's report of one run: its outcome."""

    outcome: ToolOutcome
    transition: object | None = None
    """For a `Completed` from a VoiceTask's tool, the value the host validated, which ends the
    task.
    """


ToolReturn: TypeAlias = (
    ToolReport
    | ToolError
    | ModelRetry
    | StopResponse
    | BaseException
    | Completed
    | Failed
    | Rejected
    | InputRequired
    | Working
    | ToolResult
    | object
    | None
)
"""What a tool's body returned, or the exception it raised, told apart by its class;
[`ToolReports::tool_report`] classifies it into a [`ToolOutcome`].
"""
