# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/toolsets.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import get_annotations
from typing import Any

from livekit.agents.llm import Toolset
from livekit.agents.llm.tool_context import Tool
from livekit.agents.voice.events import RunContext


@dataclass
class ToolCalledEvent:
    ctx: RunContext
    arguments: dict[str, object]


@dataclass
class ToolCompletedEvent:
    ctx: RunContext
    output: object


@dataclass
class SearchItem:
    """A search candidate derived from a single tool at index time."""

    source: Tool | Toolset
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    index_data: Any = field(default=None, repr=False)


ToolCalledEvent.__annotations__ = get_annotations(ToolCalledEvent, eval_str=True)
ToolCalledEvent.__module__ = "livekit.agents.beta"
ToolCompletedEvent.__annotations__ = get_annotations(ToolCompletedEvent, eval_str=True)
ToolCompletedEvent.__module__ = "livekit.agents.beta"
SearchItem.__annotations__ = get_annotations(SearchItem, eval_str=True)
SearchItem.__module__ = "livekit.agents.beta"
