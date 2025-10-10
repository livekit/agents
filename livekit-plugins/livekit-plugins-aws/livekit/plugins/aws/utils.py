from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents import llm
from livekit.agents.llm import FunctionTool, RawFunctionTool
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import ToolSpecificationTypeDef, ToolTypeDef
__all__ = ["to_fnc_ctx"]

DEFAULT_REGION = "us-east-1"


def to_fnc_ctx(fncs: list[FunctionTool | RawFunctionTool]) -> list[ToolTypeDef]:
    return [_build_tool_spec(fnc) for fnc in fncs]


def _build_tool_spec(function: FunctionTool | RawFunctionTool) -> ToolTypeDef:
    if is_function_tool(function):
        fnc = llm.utils.build_legacy_openai_schema(function, internally_tagged=True)
        spec: ToolSpecificationTypeDef = {
            "name": fnc["name"],
            "inputSchema": {"json": fnc["parameters"] if fnc["parameters"] else {}},
        }
        if fnc["description"]:
            spec["description"] = fnc["description"]
        return {"toolSpec": spec}
    elif is_raw_function_tool(function):
        info = get_raw_function_info(function)
        return {
            "toolSpec": {
                "name": info.name,
                "description": info.raw_schema.get("description", ""),
                "inputSchema": {"json": info.raw_schema.get("parameters", {})},
            }
        }
    else:
        raise ValueError("Invalid function tool")


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
