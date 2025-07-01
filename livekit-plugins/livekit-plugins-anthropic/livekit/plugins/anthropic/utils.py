from typing import Literal, Optional, Union

import anthropic
from livekit.agents import llm
from livekit.agents.llm import FunctionTool, RawFunctionTool
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

# We can define up to 4 cache breakpoints, we will add them at:
# - the last tool definition
# - the last system message
# - the last assistant message
# - the last user message before the last assistant message
# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#structuring-your-prompt
CACHE_CONTROL_EPHEMERAL = anthropic.types.CacheControlEphemeralParam(type="ephemeral")

__all__ = ["to_fnc_ctx", "CACHE_CONTROL_EPHEMERAL"]


def to_fnc_ctx(
    fncs: list[Union[FunctionTool, RawFunctionTool]], caching: Optional[Literal["ephemeral"]]
) -> list[anthropic.types.ToolParam]:
    tools: list[anthropic.types.ToolParam] = []
    for fnc in fncs:
        tools.append(_build_anthropic_schema(fnc))

    if tools and caching == "ephemeral":
        tools[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL

    return tools


def _build_anthropic_schema(
    function_tool: Union[FunctionTool, RawFunctionTool],
) -> anthropic.types.ToolParam:
    if is_function_tool(function_tool):
        fnc = llm.utils.build_legacy_openai_schema(function_tool, internally_tagged=True)
        return anthropic.types.ToolParam(
            name=fnc["name"],
            description=fnc["description"] or "",
            input_schema=fnc["parameters"],
        )
    elif is_raw_function_tool(function_tool):
        info = get_raw_function_info(function_tool)
        return anthropic.types.ToolParam(
            name=info.name,
            description=info.raw_schema.get("description", ""),
            input_schema=info.raw_schema.get("parameters", {}),
        )
    else:
        raise ValueError("Invalid function tool")
