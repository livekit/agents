# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/session/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import Literal, TypeAlias

from pydantic import JsonValue

from livekit.agents.llm import ChatContext

from .chat_items import ChatMessage


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallRef:
    agent_id: str
    """Agent context containing the originating tool call."""
    call_id: str
    """Canonical FunctionCall.id within that context; provider call_id values may repeat."""


AgentCallStatus: TypeAlias = Literal["approval", "ready", "running"]


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentCallState:
    origin: ToolCallRef
    """Parent tool call receiving the child's accepted output."""
    parent_invocation_id: str
    """Parent model invocation that admitted this call."""
    parent_execution_id: str
    """Parent workflow/task correlation; independent of the child agent-state ID."""
    parent_tool_name: str
    """Name of the agent-as-tool function exposed by the parent."""
    agent_name: str
    """Configured agent to execute after approval and validation."""
    validated_arguments: dict[str, JsonValue]
    """Accepted JSON arguments passed to that agent-as-tool function."""
    identity: str
    """Canonical validated-argument identity used by approval."""
    status: AgentCallStatus
    """Local admission/execution state; never an A2A task status."""
    agent_id: str | None
    """Child's entry in the session map, allocated only when it starts."""


InputRequestState: TypeAlias = Literal["open", "responding", "responded", "superseded", "closed"]


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentInputRequest:
    state: InputRequestState
    """This occurrence's response lifecycle, independent of its asking tool's completion."""
    source_message: ChatMessage
    """The requested question; native delivery receipts separately establish presentation."""
    response_message: ChatMessage | None
    """Human response reserved for this occurrence, if one was selected."""
    origin: ToolCallRef | None
    """Asking call's provenance; it never directs the next input to a tool."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentState:
    agent_name: str
    """Configured behavior; no Python Agent instance is serialized."""
    chat_ctx: ChatContext
    """This agent context's private model history."""
    origin: ToolCallRef | None
    """Parent call receiving this agent's output; None for the root."""
    calls: dict[str, AgentCallState]
    """Unresolved local agent calls, indexed by the parent's tool-call ID."""
    input_requests: dict[str, AgentInputRequest]
    """Retained input occurrences owned by this agent context."""
    completed: bool
    """Latest invocation accepted an output; a JSON null is still a completed output."""
    output: JsonValue | None
    """Accepted serialized output retained until its parent receives it."""
    error: str | None
    """Diagnostic failure, separate from a successful output or cancellation."""


ToolCallRef.__annotations__ = get_annotations(ToolCallRef, eval_str=True)
ToolCallRef.__module__ = "livekit.agents.storage.session_state"
AgentCallState.__annotations__ = get_annotations(AgentCallState, eval_str=True)
AgentCallState.__module__ = "livekit.agents.storage.session_state"
AgentInputRequest.__annotations__ = get_annotations(AgentInputRequest, eval_str=True)
AgentInputRequest.__module__ = "livekit.agents.storage.session_state"
AgentState.__annotations__ = get_annotations(AgentState, eval_str=True)
AgentState.__module__ = "livekit.agents.storage.session_state"
