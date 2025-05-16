from typing import Literal

import anthropic
from livekit.agents import llm
from livekit.agents.llm import FunctionTool

CACHE_CONTROL_EPHEMERAL = anthropic.types.CacheControlEphemeralParam(type="ephemeral")

__all__ = ["to_fnc_ctx", "CACHE_CONTROL_EPHEMERAL"]


def to_fnc_ctx(
    fncs: list[FunctionTool], caching: Literal["ephemeral"] | None
) -> list[anthropic.types.ToolParam]:
    tools: list[anthropic.types.ToolParam] = []
    for i, fnc in enumerate(fncs):
        cache_ctrl = (
            CACHE_CONTROL_EPHEMERAL if (i == len(fncs) - 1) and caching == "ephemeral" else None
        )
        tools.append(_build_anthropic_schema(fnc, cache_ctrl=cache_ctrl))

    return tools


def _build_anthropic_schema(
    function_tool: FunctionTool,
    cache_ctrl: anthropic.types.CacheControlEphemeralParam | None = None,
) -> anthropic.types.ToolParam:
    fnc = llm.utils.build_legacy_openai_schema(function_tool, internally_tagged=True)
    return anthropic.types.ToolParam(
        name=fnc["name"],
        description=fnc["description"] or "",
        input_schema=fnc["parameters"],
        cache_control=cache_ctrl,
    )
