# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/text_session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from livekit.agents._core import _core_HandlerTurn, _core_SessionBinding
    from livekit.agents.llm import LLM
    from livekit.agents.voice.agent_run import AgentCallbacks

    from .agent_run import AgentDefinition, RunFinished, RunInputRequirement, StreamEvent
    from .chat_items import ChatItem
    from .connect import APIConnectOptions
    from .session_snapshot import FrameId, _SessionSnapshot

ChatHistory: TypeAlias = Literal["fresh", "fork"]


@dataclass(kw_only=True, slots=True)
class LocalAgentTool:
    agent_name: str
    chat_history: ChatHistory
    merge_chat_ctx: bool


@dataclass(kw_only=True, slots=True)
class TextSessionAgent:
    """Shared definitions and live resources; no host resource enters the saved state."""

    definition: AgentDefinition
    llm: LLM
    host: AgentCallbacks
    output_type_explicit: bool
    agent_tools: dict[str, LocalAgentTool]
    host_context: object
    """A callback context for this agent, captured at session entry."""
    tools_host: object | None = None
    """Canonical tool pool for this agent, shared across its invocations."""


@dataclass(kw_only=True, slots=True)
class TextSessionRequest:
    output: _core_HandlerTurn | None = None
    session_sink: _core_SessionBinding | None = None
    session_id: str
    id_seed: int | None = None
    """What this request's chat item ids are drawn from; `None` draws a random one."""
    execution_id: str
    task_id: str | None = None
    message: ChatItem
    delegated: bool
    conn_options: APIConnectOptions


@dataclass(kw_only=True, slots=True)
class TextSessionFinished:
    """A settled turn as the host reads it: the active frame's run and the state to save."""

    run: RunFinished
    state: _SessionSnapshot


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateStream:
    sequence: int
    frame_id: FrameId
    agent_name: str
    invocation_id: str
    event: StreamEvent


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateInvocationSettled:
    invocation_id: str


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateParked:
    invocation_id: str


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateIncompleteCleanup:
    pending: list[str]


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateFinished:
    settled: TextSessionFinished


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateFailed:
    error: BaseException
    input_requirements: list[RunInputRequirement] = field(default_factory=list)
    provider_error: BaseException | None = None
    """The original model failure, retained beside its correlated outcome metadata."""


TextSessionUpdate: TypeAlias = (
    TextSessionUpdateStream
    | TextSessionUpdateInvocationSettled
    | TextSessionUpdateParked
    | TextSessionUpdateIncompleteCleanup
    | TextSessionUpdateFinished
    | TextSessionUpdateFailed
)


@dataclass(kw_only=True, slots=True)
class TextSessionDelivery:
    invocation_id: str
    messages: dict[str, str]
