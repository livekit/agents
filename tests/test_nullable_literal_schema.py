from typing import Literal

import pytest
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel

from livekit.agents.llm.utils import to_openai_response_format

pytestmark = pytest.mark.unit


class StringResponse(BaseModel):
    value: Literal["known"] | None


class IntegerResponse(BaseModel):
    value: Literal[1] | None


class BooleanResponse(BaseModel):
    value: Literal[True] | None


@pytest.mark.parametrize(
    "response_type, value",
    [(StringResponse, "known"), (IntegerResponse, 1), (BooleanResponse, True)],
)
def test_nullable_literal_response_schema_accepts_null_and_the_literal(
    response_type: type[BaseModel], value: object
) -> None:
    wire_schema = to_openai_response_format(response_type)["json_schema"]["schema"]
    validator = Draft202012Validator(wire_schema)
    for payload in ({"value": None}, {"value": value}):
        response_type.model_validate(payload)
        validator.validate(payload)
    with pytest.raises(ValidationError):
        validator.validate({"value": "invalid"})


class EnumResponse(BaseModel):
    value: Literal["first", "second"] | None


def test_nullable_multi_value_enum_remains_valid() -> None:
    schema = to_openai_response_format(EnumResponse)["json_schema"]["schema"]
    validator = Draft202012Validator(schema)
    for value in (None, "first", "second"):
        validator.validate({"value": value})
