from __future__ import annotations

import pytest
from pydantic import BaseModel

from livekit.agents.llm import function_tool
from livekit.agents.llm.utils import build_strict_openai_schema, to_openai_response_format

pytestmark = pytest.mark.unit


class _ResponseFormat(BaseModel):
    label: str = ""
    description: str | None = None


def test_response_format_preserves_pydantic_nullability() -> None:
    response_format = to_openai_response_format(_ResponseFormat)
    schema = response_format["json_schema"]["schema"]

    assert schema["required"] == ["label", "description"]
    assert schema["properties"]["label"]["type"] == "string"
    assert schema["properties"]["description"]["type"] == ["string", "null"]
    assert "default" not in schema["properties"]["label"]
    assert "default" not in schema["properties"]["description"]


def test_strict_tool_schema_keeps_defaulted_arguments_nullable() -> None:
    @function_tool
    async def lookup(label: str = "") -> str:
        """Look up an item by label."""
        return label

    schema = build_strict_openai_schema(lookup)["function"]["parameters"]

    assert schema["properties"]["label"]["type"] == ["string", "null"]
