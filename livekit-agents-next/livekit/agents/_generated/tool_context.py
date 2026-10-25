# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal

from typing_extensions import NotRequired, Required, TypedDict

DuplicateMode = Literal["allow", "reject", "replace", "confirm"]


DuplicateScope = Literal["name", "name_and_args"]
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


class Function(TypedDict, total=False):
    name: Required[str]


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


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
Function.__module__ = "livekit.agents.llm.tool_context"
RawFunctionDescription.__module__ = "livekit.agents.llm.tool_context"
