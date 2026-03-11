from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from livekit.agents.llm._strict import to_strict_json_schema


class NullableEnumModel(BaseModel):
    status: Literal["active", "inactive"] | None = Field(None)


class NullableBoolModel(BaseModel):
    flag: bool | None = Field(None)


class NonNullableEnumModel(BaseModel):
    status: Literal["active", "inactive"] = Field(...)


def test_nullable_enum_includes_null_in_enum():
    schema = to_strict_json_schema(NullableEnumModel)
    status = schema["properties"]["status"]
    assert None in status["enum"], f"enum should contain None: {status}"
    assert "null" in status["type"], f"type should contain 'null': {status}"


def test_nullable_bool_has_null_type():
    schema = to_strict_json_schema(NullableBoolModel)
    flag = schema["properties"]["flag"]
    assert "enum" not in flag, f"bool field should not have enum: {flag}"
    assert "null" in flag["type"], f"type should contain 'null': {flag}"


def test_non_nullable_enum_excludes_null():
    schema = to_strict_json_schema(NonNullableEnumModel)
    status = schema["properties"]["status"]
    assert None not in status["enum"], f"enum should not contain None: {status}"
    assert "null" not in status.get("type", []), f"type should not contain 'null': {status}"
