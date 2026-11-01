# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing import NotRequired


# Record declarations:


DuplicateMode: TypeAlias = Literal["allow", "reject", "replace", "confirm"]


DuplicateScope: TypeAlias = Literal["name", "name_and_args"]
"""What counts as a duplicate of an in-flight call, i.e. what ``on_duplicate`` acts on.

``"name"``           any in-flight call of the same tool (default).
``"name_and_args"``  same tool *and* same arguments, so concurrent calls of one
                     tool with different arguments are not duplicates::

                         @function_tool(on_duplicate="reject", duplicate_scope="name_and_args")
                         async def check_order(ctx: RunContext, order_id: str) -> str: ...

                     Arguments are compared *after* validation, so a parameter the
                     LLM omitted on one call and passed explicitly on the next still
                     reads as the same call, and ``1`` matches ``1.0`` for a ``float``
                     parameter. Raw function tools (including MCP tools) have no
                     per-parameter schema, so their arguments are compared as sent.

                     Comparison fails open — an unrepresentable argument or a
                     validation error leaves the call treated as sent rather than
                     blocking it — so don't rely on this as an exactly-once
                     guarantee; put that in the tool body.
"""


class RawFunctionDescription(TypedDict):
    """Represents the raw function schema format used in LLM function calling APIs.

    This structure directly maps to OpenAI's function definition format as documented at:
    https://platform.openai.com/docs/guides/function-calling?api-mode=responses

    It is also compatible with other LLM providers that support raw JSON Schema-based
    function definitions.
    """

    name: str
    description: NotRequired[str | None]
    parameters: dict[str, object]


RawFunctionDescription.__module__ = "livekit.agents.llm.tool_context"


CONFIRM_DUPLICATE_DESCRIPTION: Final = "Set this to True to confirm you want to run a duplicate. Only do this when user confirms the duplication is needed."
"""The description the model reads for that parameter. A prompt, so rewording it changes
behaviour: it must say the user asked for the repeat.
"""
