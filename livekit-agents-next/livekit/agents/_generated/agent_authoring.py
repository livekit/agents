# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/agent/authoring.rs

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from inspect import get_annotations
from typing import TYPE_CHECKING, Any, Final

from typing_extensions import TypedDict

from livekit.agents.llm.tool_context import ToolChoice

from ..llm.tool_context import ReadBack
from ..types import NOT_GIVEN, NotGivenOr
from .tool_context import DuplicateMode, DuplicateScope, RawFunctionDescription
from .tool_flags import ToolFlag

if TYPE_CHECKING:
    from .agent_run import Retries


# Record declarations:


class AgentRetries(TypedDict, total=False):
    """A partial retry budget; omitted keys inherit the agent's configured budget."""

    tools: int
    output: int


@dataclass
class ModelSettings:
    tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN


@dataclass
class ToolOptions:
    """`Agent.tool()`'s options; an option left out keeps its default."""

    name: NotGivenOr[str | None] = NOT_GIVEN
    """The tool's name; the function's when `None`."""
    description: NotGivenOr[str | None] = NOT_GIVEN
    """What the model reads; the docstring when `None`."""
    raw_schema: NotGivenOr[RawFunctionDescription | dict[str, Any] | None] = NOT_GIVEN
    """The tool's whole schema, for a function that takes its arguments as one raw dict."""
    flags: NotGivenOr[ToolFlag] = NOT_GIVEN
    on_duplicate: NotGivenOr[DuplicateMode] = NOT_GIVEN
    duplicate_scope: NotGivenOr[DuplicateScope] = NOT_GIVEN
    needs_approval: NotGivenOr[bool | ReadBack] = NOT_GIVEN
    """Hold each call for the person's approval of a read-back (tools.md "Approval"): `True`
    reads back the description and the arguments, a `ReadBack` function writes it.
    """
    retries: NotGivenOr[int | None] = NOT_GIVEN
    """Consecutive retries the model gets for this tool; the agent's `retries['tools']` when
    `None`.
    """


class ToolOptionsKwargs(TypedDict, total=False):
    """`ToolOptions`'s fields as keyword arguments."""

    name: str | None
    """The tool's name; the function's when `None`."""
    description: str | None
    """What the model reads; the docstring when `None`."""
    raw_schema: RawFunctionDescription | dict[str, Any] | None
    """The tool's whole schema, for a function that takes its arguments as one raw dict."""
    flags: ToolFlag
    on_duplicate: DuplicateMode
    duplicate_scope: DuplicateScope
    needs_approval: bool | ReadBack
    """Hold each call for the person's approval of a read-back (tools.md "Approval"): `True`
    reads back the description and the arguments, a `ReadBack` function writes it.
    """
    retries: int | None
    """Consecutive retries the model gets for this tool; the agent's `retries['tools']` when
    `None`.
    """


@unique
class CallbackKind(StrEnum):
    TOOL = "tool"
    INSTRUCTIONS = "instructions"
    VALIDATOR = "validator"
    ON_ENTER = "on_enter"
    ON_EXIT = "on_exit"


class AgentConfiguration(TypedDict):
    """Native configuration combined with SDK-derived tool and output schemas at the binding."""

    name: str
    instructions: str
    dynamic_instructions: bool
    retries: Retries


@dataclass(kw_only=True, slots=True)
class Registered:
    """A callback's registration: its `token`, and the one it replaced (`retired`), an `on_enter` or
    `on_exit` registered again.
    """

    token: int
    retired: int | None


AgentRetries.__module__ = "livekit.agents.voice"
ModelSettings.__annotations__ = get_annotations(ModelSettings, eval_str=True)
ModelSettings.__module__ = "livekit.agents.voice"


DEFAULT_AGENT_NAME: Final = "default_agent"
