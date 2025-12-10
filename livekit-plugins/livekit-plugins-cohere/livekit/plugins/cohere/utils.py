from typing import Union

from livekit.agents import llm
from livekit.agents.llm import FunctionTool, RawFunctionTool
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

__all__ = ["to_fnc_ctx"]


def to_fnc_ctx(fncs: list[Union[FunctionTool, RawFunctionTool]]) -> list[dict]:
    """Convert function tools to Cohere tool format."""
    tools: list[dict] = []
    for fnc in fncs:
        tools.append(_build_cohere_schema(fnc))
    return tools


def _build_cohere_schema(
    function_tool: Union[FunctionTool, RawFunctionTool],
) -> dict:
    """Build Cohere tool schema from function tool."""
    if is_function_tool(function_tool):
        fnc = llm.utils.build_legacy_openai_schema(function_tool, internally_tagged=True)
        return {
            "name": fnc["name"],
            "description": fnc["description"] or "",
            "parameter_definitions": fnc["parameters"].get("properties", {}),
        }
    elif is_raw_function_tool(function_tool):
        info = get_raw_function_info(function_tool)
        return {
            "name": info.name,
            "description": info.raw_schema.get("description", ""),
            "parameter_definitions": info.raw_schema.get("parameters", {}).get("properties", {}),
        }
    else:
        raise ValueError("Invalid function tool")
