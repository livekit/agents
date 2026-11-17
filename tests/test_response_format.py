from __future__ import annotations

from typing import Literal

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel

from livekit.agents.llm.utils import (
    _json_schema_allows_null,  # null-acceptance checker; asserts the wire schema
    to_openai_response_format,
)

pytestmark = pytest.mark.unit


def _wire_property(response_format: type, field: str) -> tuple[dict, dict]:
    schema = to_openai_response_format(response_format)["json_schema"]["schema"]
    return schema["properties"][field], schema


class _ResponseFormat(BaseModel):
    label: str = "default"
    description: str | None = None


def test_response_format_drops_defaults_and_keeps_fields_required() -> None:
    response_format = to_openai_response_format(_ResponseFormat)
    schema = response_format["json_schema"]["schema"]

    assert schema["required"] == ["label", "description"]
    # a defaulted non-nullable field stays non-nullable: the model must produce
    # a value, and plain pydantic validation accepts whatever it returns
    assert schema["properties"]["label"]["type"] == "string"
    assert "default" not in schema["properties"]["label"]
    # a genuinely nullable field keeps accepting null
    assert _json_schema_allows_null(schema["properties"]["description"], root=schema)


def test_response_format_literal_default_not_nullable() -> None:
    class Response(BaseModel):
        label: Literal["a", "b"] = "a"

    prop, root = _wire_property(Response, "label")
    assert not _json_schema_allows_null(prop, root=root)
    assert "default" not in prop


def test_response_format_union_default_not_nullable() -> None:
    class Response(BaseModel):
        value: int | str = 5

    prop, root = _wire_property(Response, "value")
    assert not _json_schema_allows_null(prop, root=root)


def test_response_format_nested_model_default_not_nullable() -> None:
    class Child(BaseModel):
        x: int = 1

    class Response(BaseModel):
        child: Child = Child()

    prop, root = _wire_property(Response, "child")
    assert not _json_schema_allows_null(prop, root=root)


def test_response_format_genuine_null_via_enum_preserved() -> None:
    # nullability expressed through enum membership is real nullability, not a default
    class Response(BaseModel):
        value: Literal["x", None] = "x"

    prop, root = _wire_property(Response, "value")
    assert _json_schema_allows_null(prop, root=root)


def test_response_format_payload_validates_with_plain_model_validate() -> None:
    # the decode contract: a schema-conforming payload passes direct validation,
    # with no sentinel resolution step
    class Child(BaseModel):
        x: int = 1

    class Response(BaseModel):
        child: Child = Child()
        label: str = "a"

    payload = {"child": {"x": 2}, "label": "b"}
    assert Response.model_validate(payload) == Response(child=Child(x=2), label="b")


class _StringResponse(BaseModel):
    value: Literal["known"] | None


class _IntegerResponse(BaseModel):
    value: Literal[1] | None


class _BooleanResponse(BaseModel):
    value: Literal[True] | None


class _EnumResponse(BaseModel):
    value: Literal["first", "second"] | None


@pytest.mark.parametrize(
    ("response_type", "values"),
    [
        (_StringResponse, ["known"]),
        (_IntegerResponse, [1]),
        (_BooleanResponse, [True]),
        (_EnumResponse, ["first", "second"]),
    ],
)
def test_nullable_literal_response_schema_accepts_null_and_the_literals(
    response_type: type[BaseModel], values: list[object]
) -> None:
    wire_schema = to_openai_response_format(response_type)["json_schema"]["schema"]
    validator = Draft202012Validator(wire_schema)
    for payload in ({"value": v} for v in [None, *values]):
        response_type.model_validate(payload)
        validator.validate(payload)
    with pytest.raises(ValidationError):
        validator.validate({"value": "invalid"})
