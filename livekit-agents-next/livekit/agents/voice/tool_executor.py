"""The tool handling options of an agent: the templates rendered around background tool calls
(``ctx.update()``, duplicate handling, the deferred reply), set on the ``Agent`` and overridden by
an ``AsyncToolset`` for its own tools. Calls themselves run through the agent's ``AgentHost``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

from .._generated.tool_executor import AsyncToolOptions, ToolHandlingOptions

if TYPE_CHECKING:
    from ..llm.tool_context import Tool, Toolset

__all__ = ["AsyncToolOptions", "ToolHandlingOptions", "UpdatePromptArgs"]

UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""


class UpdatePromptArgs(TypedDict):
    """Args for the ``update`` template."""

    function_name: str
    call_id: str
    message: str


class DuplicatePromptArgs(TypedDict):
    """Args for the ``duplicate_reject`` / ``duplicate_confirm`` templates."""

    function_name: str
    fnc_calls_json: list[str]
    fnc_calls_text: str


class ReplyPromptArgs(TypedDict):
    """Args for the ``reply_at_tail`` / ``reply_maybe_covered`` templates."""

    call_ids: list[str]


def render_template(
    agent_options: ToolHandlingOptions | None,
    tools: Sequence[Tool | Toolset],
    tool: str | None,
    key: str,
    args: dict[str, Any],
) -> str | None:
    """The ``key`` template rendered with ``args``: the innermost ``AsyncToolset`` among
    ``tools`` holding ``tool`` that sets it, else the agent's; ``None`` when neither does."""
    template = _toolset_template(tools, tool, key) if tool is not None else None
    if template is None:
        template = _template(agent_options, key)
    if template is None:
        return None
    return str(template(args) if callable(template) else template.format(**args))


def _toolset_template(tools: Sequence[Tool | Toolset], tool: str, key: str) -> Any:
    from ..llm.tool_context import AsyncToolset, ToolContext, Toolset

    for entry in tools:
        if not isinstance(entry, Toolset):
            continue
        template = _toolset_template(entry.tools, tool, key)
        if (
            template is None
            and isinstance(entry, AsyncToolset)
            and ToolContext([entry]).get_function_tool(tool) is not None
        ):
            template = _template(entry._tool_handling, key)
        if template is not None:
            return template
    return None


def _template(options: ToolHandlingOptions | None, key: str) -> Any:
    templates: Mapping[str, Any] = (options or {}).get("async_options") or {}
    return templates.get(key)
