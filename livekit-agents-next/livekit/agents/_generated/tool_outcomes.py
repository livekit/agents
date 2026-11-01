# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import get_annotations


@dataclass(frozen=True)
class Failed:
    """The task failed: ``error`` is its last message."""

    error: str


@dataclass(frozen=True)
class Rejected:
    """The task will not run this input, from ``on_enter``."""

    reason: str


@dataclass(frozen=True)
class Working:
    """The task waits on a system: nothing is delivered, ``message`` is its status."""

    message: str | None = None


@dataclass
class ToolResult:
    """A function tool's output, whether it requests a follow-up reply, and whether the person
        gets it as written.

        Args:
            output: The value to return to the model, such as a string, dict, list,
                or None. For a handoff, return ``(agent, ToolResult(...))``.
            reply_required: Whether this result requests a reply. Defaults to True.
                False keeps the output in chat history without requesting a reply.
                Other tool results can still request a reply, and interruption or
                ``cancel_tool_reply()`` can still prevent one.
            verbatim: The output is text the person gets exactly as written: spoken with ``say()``
                on a call, sent unchanged in text, relayed unchanged by a delegate. With
                ``reply_required=False`` it ends the turn; otherwise the model continues after it.

        Example:
            return ToolResult("Successfully sent DTMF events: 1", reply_required=False)
    """

    output: object
    reply_required: bool = field(default=True, kw_only=True)
    verbatim: bool = field(default=False, kw_only=True)


Failed.__annotations__ = get_annotations(Failed, eval_str=True)
Failed.__module__ = "livekit.agents.llm.tool_context"
Rejected.__annotations__ = get_annotations(Rejected, eval_str=True)
Rejected.__module__ = "livekit.agents.llm.tool_context"
Working.__annotations__ = get_annotations(Working, eval_str=True)
Working.__module__ = "livekit.agents.llm.tool_context"
ToolResult.__annotations__ = get_annotations(ToolResult, eval_str=True)
ToolResult.__module__ = "livekit.agents.llm.tool_context"
