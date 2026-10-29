"""The tool handling options of a voice session: the templates rendered around background tool
calls (``ctx.update()``, duplicate handling, the deferred reply). Calls themselves run through the
agent's ``AgentHost``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import TypedDict

from .._generated.tool_executor import AsyncToolOptions, ToolHandlingOptions

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


def _render(template: str | Callable[[Any], str], args: dict[str, Any]) -> str:
    return template(args) if callable(template) else template.format(**args)
