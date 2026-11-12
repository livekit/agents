"""Phonic realtime plugin: tool definitions."""

from __future__ import annotations

from livekit.agents import llm
from livekit.plugins.phonic.realtime import to_phonic_tool_definitions


def test_tools_reach_phonic_as_strict_json_schema_definitions() -> None:
    """A function tool reaches Phonic as a strict JSON-schema tool definition."""

    @llm.function_tool(
        name="search_pizza_shop_recs",
        description="Search for pizza shop recommendations in a location.",
    )
    async def search(location: str) -> str:
        return location

    @llm.function_tool(name="undescribed")
    async def undescribed() -> None:
        pass

    definitions = to_phonic_tool_definitions(llm.ToolContext([search, undescribed]))
    assert definitions[1]["description"] == ""
    assert definitions[:1] == [
        {
            "name": "search_pizza_shop_recs",
            "description": "Search for pizza shop recommendations in a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    ]
