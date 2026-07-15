from __future__ import annotations

import pytest
from pydantic import BaseModel

from livekit.agents.llm.utils import to_openai_response_format, validate_response_format

pytestmark = pytest.mark.unit


class _ResponseFormat(BaseModel):
    label: str = "default"
    description: str | None = None


def test_response_format_encodes_default_as_nullable() -> None:
    response_format = to_openai_response_format(_ResponseFormat)
    schema = response_format["json_schema"]["schema"]

    assert schema["required"] == ["label", "description"]
    assert schema["properties"]["label"]["type"] == ["string", "null"]
    assert schema["properties"]["description"]["type"] == ["string", "null"]
    assert "default" not in schema["properties"]["label"]
    assert "default" not in schema["properties"]["description"]


def test_validate_response_format_injects_defaults_for_non_nullable_fields() -> None:
    response = validate_response_format(
        _ResponseFormat,
        {"label": None, "description": None},
    )

    assert response == _ResponseFormat(label="default", description=None)


def test_validate_response_format_injects_nested_defaults() -> None:
    class Item(BaseModel):
        color: str = "red"
        note: str | None = "fallback"

    class Response(BaseModel):
        items: list[Item]

    response = validate_response_format(
        Response,
        {"items": [{"color": None, "note": None}]},
    )

    assert response == Response(items=[Item(color="red", note=None)])
