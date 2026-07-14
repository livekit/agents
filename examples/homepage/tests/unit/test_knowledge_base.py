import pytest
from knowledge_base import KnowledgeBase

from livekit.agents.llm.tool_context import get_raw_function_info

pytestmark = pytest.mark.unit

EXPECTED_TOPICS = {
    "agent-builder",
    "agent-observability",
    "agents-on-livekit-cloud",
    "livekit-inference",
    "livekit-phone-numbers",
    "platform",
}


def test_lookup_tool_index_is_built_from_the_knowledge_base() -> None:
    schema = get_raw_function_info(KnowledgeBase().lookup_tool()).raw_schema

    assert schema["name"] == "lookup_product"
    assert set(schema["parameters"]["properties"]["product"]["enum"]) == EXPECTED_TOPICS
    for topic in EXPECTED_TOPICS:
        assert f"- {topic}: " in schema["description"]
