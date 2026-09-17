import pytest

from livekit.agents import llm
from livekit.plugins.phonic.realtime import to_phonic_tool_definitions

pytestmark = pytest.mark.unit


def test_to_phonic_tool_definitions() -> None:
    @llm.function_tool(
        name="search_pizza_shop_recs",
        description="Search for pizza shop recommendations in a location.",
    )
    async def search_pizza_shop_recs(location: str) -> str:
        return location

    definitions = to_phonic_tool_definitions(llm.ToolContext([search_pizza_shop_recs]))

    assert definitions == [
        {
            "name": "search_pizza_shop_recs",
            "description": "Search for pizza shop recommendations in a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False,
            },
        }
    ]


def test_to_phonic_tool_definitions_allows_no_description() -> None:
    @llm.function_tool(name="search_pizza_shop_recs")
    async def search_pizza_shop_recs() -> None:
        pass

    assert (
        to_phonic_tool_definitions(llm.ToolContext([search_pizza_shop_recs]))[0]["description"]
        == ""
    )
