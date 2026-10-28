from __future__ import annotations

import pytest

from livekit.agents import llm
from livekit.plugins.xai.tools import (
    FileSearch,
    WebSearch,
    XSearch,
    _raise_if_xai_tool_reserved_name_conflict,
)

pytestmark = pytest.mark.unit


def _named(name: str) -> llm.FunctionTool:
    @llm.function_tool(name=name)
    async def tool() -> str:
        return "ok"

    return tool


def _raw(name: str) -> llm.RawFunctionTool:
    @llm.function_tool(
        raw_schema={
            "name": name,
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        }
    )
    async def tool() -> str:
        return "ok"

    return tool


@llm.function_tool
async def collections_search() -> str:
    """A function that collides with xAI FileSearch."""
    return "hit"


_RESERVED_PAIRS: list[tuple[llm.ProviderTool, str]] = [
    (WebSearch(), "web_search"),
    (WebSearch(), "browse_page"),
    (XSearch(), "x_keyword_search"),
    (XSearch(), "x_semantic_search"),
    (XSearch(), "x_user_search"),
    (XSearch(), "x_thread_fetch"),
    (FileSearch(), "collections_search"),
    (FileSearch(), "file_search"),
]


@pytest.mark.parametrize(
    ("provider_tool", "function_name"),
    _RESERVED_PAIRS,
    ids=[f"{t.__class__.__name__}-{name}" for t, name in _RESERVED_PAIRS],
)
def test_provider_tool_plus_reserved_function_raises(
    provider_tool: llm.ProviderTool, function_name: str
) -> None:
    with pytest.raises(ValueError, match="Rename or remove"):
        _raise_if_xai_tool_reserved_name_conflict([provider_tool, _named(function_name)])


def test_file_search_plus_plain_collections_search_raises() -> None:
    with pytest.raises(ValueError, match="Rename or remove"):
        _raise_if_xai_tool_reserved_name_conflict([FileSearch(), collections_search])


def test_file_search_plus_raw_collections_search_raises() -> None:
    with pytest.raises(ValueError, match="Rename or remove"):
        _raise_if_xai_tool_reserved_name_conflict([FileSearch(), _raw("collections_search")])


def test_file_search_plus_nested_toolset_reserved_name_raises() -> None:
    toolset = llm.Toolset(id="nested", tools=[collections_search])
    tools = llm.ToolContext([FileSearch(), toolset]).flatten()
    with pytest.raises(ValueError, match="Rename or remove"):
        _raise_if_xai_tool_reserved_name_conflict(tools)


_HARMLESS_PAIRS: list[tuple[llm.ProviderTool | None, str]] = [
    (WebSearch(), "view_image"),
    (WebSearch(), "harmless_control"),
    (WebSearch(), "collections_search"),
    (XSearch(), "x_search"),
    (XSearch(), "harmless_control"),
    (FileSearch(), "view_document"),
    (FileSearch(), "harmless_control"),
    (FileSearch(), "web_search"),
    (None, "collections_search"),
    (None, "file_search"),
    (None, "web_search"),
    (None, "browse_page"),
]


@pytest.mark.parametrize(
    ("provider_tool", "function_name"),
    _HARMLESS_PAIRS,
    ids=[
        f"{t.__class__.__name__ if t is not None else 'none'}-{name}" for t, name in _HARMLESS_PAIRS
    ],
)
def test_non_reserved_combinations_are_ok(
    provider_tool: llm.ProviderTool | None, function_name: str
) -> None:
    tools: list[llm.Tool] = [_named(function_name)]
    if provider_tool is not None:
        tools.insert(0, provider_tool)
    _raise_if_xai_tool_reserved_name_conflict(tools)


def test_file_search_alone_is_ok() -> None:
    _raise_if_xai_tool_reserved_name_conflict([FileSearch()])
