from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from livekit.agents.llm.utils import (
    _json_schema_allows_null,  # verified null-acceptance checker; asserts the wire schema
    to_openai_response_format,
    validate_response_format,
)

pytestmark = pytest.mark.unit


def _wire_property(response_format: type, field: str) -> tuple[dict, dict]:
    schema = to_openai_response_format(response_format)["json_schema"]["schema"]
    return schema["properties"][field], schema


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
    class PrimaryItem(BaseModel):
        kind: Literal["primary"]
        color: str = "red"
        note: str | None = "fallback"

    class SecondaryItem(BaseModel):
        kind: Literal["secondary"]
        color: str = "blue"

    class Response(BaseModel):
        items: list[PrimaryItem | SecondaryItem]

    response = validate_response_format(
        Response,
        {"items": [{"kind": "primary", "color": None, "note": None}]},
    )

    assert response == Response(items=[PrimaryItem(kind="primary", color="red", note=None)])


def test_response_format_encodes_literal_default_as_nullable() -> None:
    # a Literal carries const/enum, which reject null even when the type lists it
    class Response(BaseModel):
        label: Literal["a", "b"] = "a"

    prop, root = _wire_property(Response, "label")
    assert _json_schema_allows_null(prop, root=root)
    assert validate_response_format(Response, {"label": None}).label == "a"


def test_response_format_encodes_union_default_as_nullable() -> None:
    # a top-level anyOf has no direct type to append null to
    class Response(BaseModel):
        value: int | str = 5

    prop, root = _wire_property(Response, "value")
    assert _json_schema_allows_null(prop, root=root)
    assert validate_response_format(Response, {"value": None}).value == 5


def test_response_format_encodes_nested_model_default_as_nullable() -> None:
    # a defaulted nested model is a $ref with a sibling default
    class Child(BaseModel):
        x: int = 1

    class Response(BaseModel):
        child: Child = Child()

    prop, root = _wire_property(Response, "child")
    assert _json_schema_allows_null(prop, root=root)
    assert validate_response_format(Response, {"child": None}).child == Child(x=1)


def test_validate_response_format_injects_defaults_in_dict_values() -> None:
    class Child(BaseModel):
        x: int = 1

    class Response(BaseModel):
        children: dict[str, Child]

    response = validate_response_format(Response, {"children": {"k": {"x": None}}})

    assert response == Response(children={"k": Child(x=1)})


def test_validate_response_format_injects_defaults_in_tuple_items() -> None:
    class Child(BaseModel):
        x: int = 1

    class Response(BaseModel):
        pair: tuple[str, Child]

    response = validate_response_format(Response, {"pair": ["k", {"x": None}]})

    assert response == Response(pair=("k", Child(x=1)))


def test_validate_response_format_preserves_genuine_null_declared_via_enum() -> None:
    # nullability can be expressed through enum membership instead of type
    class Response(BaseModel):
        value: Literal["x", None] = "x"

    response = validate_response_format(Response, {"value": None})

    assert response.value is None


def test_validate_response_format_injects_default_when_enum_excludes_null() -> None:
    class Response(BaseModel):
        value: Literal["x", "y"] = "x"

    response = validate_response_format(Response, {"value": None})

    assert response.value == "x"


def test_validate_response_format_selects_union_variant_by_payload_keys() -> None:
    class PointsA(BaseModel):
        common: int
        a: int = 1

    class PointsB(BaseModel):
        common: int
        b: str = "bee"

    class Response(BaseModel):
        item: PointsA | PointsB

    # the payload carries PointsB's field; PointsA must not win just because
    # its only required field is present and the extra key is ignored
    response = validate_response_format(Response, {"item": {"common": 7, "b": None}})

    assert response == Response(item=PointsB(common=7, b="bee"))
