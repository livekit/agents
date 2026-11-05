# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/agent/authoring.rs

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from inspect import get_annotations
from typing import TYPE_CHECKING, Final

from typing_extensions import TypedDict

from livekit.agents.llm.tool_context import ToolChoice

from ..types import NOT_GIVEN, NotGivenOr

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


AgentRetries.__module__ = "livekit.agents.voice"
ModelSettings.__annotations__ = get_annotations(ModelSettings, eval_str=True)
ModelSettings.__module__ = "livekit.agents.voice"


DEFAULT_AGENT_NAME: Final = "default_agent"
