from __future__ import annotations

import os
from collections.abc import Awaitable
from typing import Callable, Union

from livekit.agents import llm
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from openai.types.chat import ChatCompletionToolParam

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


def to_fnc_ctx(
    fnc_ctx: list[llm.FunctionTool | llm.RawFunctionTool],
) -> list[ChatCompletionToolParam]:
    tools: list[ChatCompletionToolParam] = []
    for fnc in fnc_ctx:
        if is_raw_function_tool(fnc):
            info = get_raw_function_info(fnc)
            tools.append(
                {
                    "type": "function",
                    "function": info.raw_schema,  # type: ignore
                }
            )
        elif is_function_tool(fnc):
            tools.append(llm.utils.build_strict_openai_schema(fnc))  # type: ignore

    return tools
