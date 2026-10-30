"""The tool handling options of an agent: the templates rendered around background tool calls
(``ctx.update()``, duplicate handling, the deferred reply), set on the ``Agent`` and overridden by
an ``AsyncToolset`` for its own tools. Calls themselves run through the agent's ``AgentCallbacks``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

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


class AsyncToolOptions(TypedDict, total=False):
    """System-message templates injected around async tool dispatch.

    Each field is either a ``str.format()`` template or a callable returning a string,
    with the args typed as ``UpdatePromptArgs`` / ``DuplicatePromptArgs`` /
    ``ReplyPromptArgs``. Unmentioned keys keep their defaults.
    """

    update_template: str | Callable[[UpdatePromptArgs], str]
    """Wraps a user-provided ``ctx.update(message)`` string before it lands in chat_ctx."""
    duplicate_reject_template: str | Callable[[DuplicatePromptArgs], str]
    """Sent to the LLM when ``on_duplicate='reject'`` blocks a duplicate call."""
    duplicate_confirm_template: str | Callable[[DuplicatePromptArgs], str]
    """Sent to the LLM when ``on_duplicate='confirm'`` requires re-call with confirmation."""
    reply_at_tail_template: str | Callable[[ReplyPromptArgs], str]
    """Instruction for the deferred reply when the pending update is still the tail of chat_ctx."""
    reply_maybe_covered_template: str | Callable[[ReplyPromptArgs], str]
    """Instruction for the deferred reply when newer items came after the pending update."""


class ToolHandlingOptions(TypedDict, total=False):
    """Configuration for the tool handling system.

    Can be passed as a plain dict::

        Agent(
            instructions="...",
            tool_handling={
                "async_options": {"update_template": "..."},
            },
        )

    Set on ``Agent``, or on an ``AsyncToolset`` for its own tools (most specific wins).
    """

    async_options: AsyncToolOptions
    """Templates injected around async tool dispatch (``ctx.update()``, duplicate
    handling, coalesced replies). Unmentioned keys keep their defaults.
    """


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
