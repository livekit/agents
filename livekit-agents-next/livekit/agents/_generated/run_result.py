# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/agent/run.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from inspect import get_annotations
from typing import TYPE_CHECKING, Any, Generic, Literal

from typing_extensions import TypeAliasType, TypeVar

from .agent_run import RunInputRequirement
from .chat_items import ChatItem
from .storage_turn import PendingCall
from .usage import Usage

if TYPE_CHECKING:
    from livekit.agents.voice.run_result import RunAssert

Output_T = TypeVar("Output_T", default=Any)


@dataclass(kw_only=True)
class _Settled:
    """What every branch of the caller's `RunResult` exposes: its history, usage and correlated
    input requests. The host builds a branch from a `RunSettled`.
    """

    _new_messages: list[ChatItem] = field(repr=False)
    _all_messages: list[ChatItem] = field(repr=False)
    usage: Usage = field(repr=False)
    """What this invocation consumed: an immutable, settled snapshot (usage.md)."""
    verbatim: str | None = field(repr=False)
    """The part of the answer, or of the question, that reached the person exactly as written;
    `None` when nothing did (tools.md "Verbatim").
    """
    _pending_call: PendingCall | None = field(default=None, repr=False)
    input_requirements: list[RunInputRequirement] = field(default_factory=list, repr=False)
    """All correlated tool requests awaiting input, in model order. This metadata does not select
    a presentation order or answer additional requests automatically.
    """

    def new_messages(self) -> list[ChatItem]:
        """This invocation's input and everything it recorded, tool calls and results included."""
        from .. import _bindings

        return _bindings.run_new_messages(self)

    def all_messages(self) -> list[ChatItem]:
        """The supplied history plus this invocation, without instructions: the next direct
        run's ``message_history``.
        """
        from .. import _bindings

        return _bindings.run_all_messages(self)

    @functools.cached_property
    def expect(self) -> RunAssert:
        """Assertions over this invocation's events, in order (testing.md)."""
        from .. import _bindings

        return _bindings.run_expect(self)


@dataclass(kw_only=True)
class RunCompleted(_Settled, Generic[Output_T]):
    """`RunResult` with `status == "completed"`: `output` is the accepted value."""

    output: Output_T
    status: Literal["completed"] = "completed"


@dataclass(kw_only=True)
class RunInputRequired(_Settled):
    """`RunResult` with `status == "input_required"`: the invocation asks `question`."""

    question: str
    status: Literal["input_required"] = "input_required"


@dataclass(kw_only=True)
class RunNoOutput(_Settled):
    """`RunResult` with `status == "no_output"`: the turn ended without an output, on
    `StopResponse`, a tool result asking for no reply, a silent step or an interruption.
    """

    status: Literal["no_output"] = "no_output"


RunResult = TypeAliasType(
    "RunResult",
    RunCompleted[Output_T] | RunInputRequired | RunNoOutput,
    type_params=(Output_T,),
)
"""One agent invocation (agents.md "Running a turn"): narrow on `status` for `output` or
`question`; every branch exposes `new_messages()`, `all_messages()` and `usage`.
"""


RunCompleted.__annotations__ = get_annotations(RunCompleted, eval_str=True)
RunCompleted.__module__ = "livekit.agents.voice.run_result"
RunInputRequired.__annotations__ = get_annotations(RunInputRequired, eval_str=True)
RunInputRequired.__module__ = "livekit.agents.voice.run_result"
RunNoOutput.__annotations__ = get_annotations(RunNoOutput, eval_str=True)
RunNoOutput.__module__ = "livekit.agents.voice.run_result"
