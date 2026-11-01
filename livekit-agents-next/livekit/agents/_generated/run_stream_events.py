# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/agent/stream.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from .agent_run import RunUpdate
from .text_session import TextSessionUpdate

if TYPE_CHECKING:
    from typing import Literal

    from pydantic import JsonValue


@dataclass(frozen=True, kw_only=True, slots=True)
class TextDeltaEvent:
    sequence: int
    agent_name: str
    invocation_id: str
    frame_id: str | None = None
    message_id: str
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, kw_only=True, slots=True)
class TextEndEvent:
    sequence: int
    agent_name: str
    invocation_id: str
    frame_id: str | None = None
    message_id: str
    interrupted: bool
    type: Literal["text_end"] = "text_end"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallEvent:
    sequence: int
    agent_name: str
    invocation_id: str
    frame_id: str | None = None
    call_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolResultEvent:
    sequence: int
    agent_name: str
    invocation_id: str
    frame_id: str | None = None
    call_id: str
    tool_name: str
    outcome: Literal['completed', 'failed', 'cancelled']
    output: JsonValue
    error: str | None
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, kw_only=True, slots=True)
class ProgressEvent:
    sequence: int
    agent_name: str
    invocation_id: str
    frame_id: str | None = None
    message: str
    call_id: str | None
    tool_name: str | None
    type: Literal["progress"] = "progress"


PresentedEvent: TypeAlias = (
    TextDeltaEvent | TextEndEvent | ToolCallEvent | ToolResultEvent | ProgressEvent
)


StreamOutcome: TypeAlias = RunUpdate | TextSessionUpdate
"""Only terminal updates cross to the host's result hydration callbacks."""


TextDeltaEvent.__module__ = "livekit.agents.voice.run_stream"
TextEndEvent.__module__ = "livekit.agents.voice.run_stream"
ToolCallEvent.__module__ = "livekit.agents.voice.run_stream"
ToolResultEvent.__module__ = "livekit.agents.voice.run_stream"
ProgressEvent.__module__ = "livekit.agents.voice.run_stream"
