import copy
from typing import Literal

import pytest
from jsonschema import validate
from pydantic import BaseModel

from livekit.agents.llm import function_tool
from livekit.agents.llm.utils import build_strict_openai_schema, prepare_function_arguments

pytestmark = pytest.mark.unit


class EmailRequest(BaseModel):
    kind: Literal["email"]
    retries: int = 2
    note: str | None = "email default"


class SMSRequest(BaseModel):
    kind: Literal["sms"]
    retries: int = 5
    note: str | None = "sms default"


@pytest.mark.parametrize("shape", ["list", "nested_list", "tuple", "variadic_tuple"])
@pytest.mark.parametrize("kind,retries", [("email", 2), ("sms", 5)])
async def test_container_union_uses_matching_variant_defaults(shape, kind, retries):
    # Both branches have the same fields, so applying the wrong branch's defaults
    # still validates but silently changes the value the tool receives.
    item = {"kind": kind, "retries": None, "note": None}
    if shape == "list":

        @function_tool
        async def tool(requests: list[EmailRequest] | list[SMSRequest]) -> int:
            return requests[0].retries

        requests = [item]
    elif shape == "nested_list":

        @function_tool
        async def tool(requests: list[list[EmailRequest]] | list[list[SMSRequest]]) -> int:
            return requests[0][0].retries

        requests = [[item]]
    elif shape == "tuple":

        @function_tool
        async def tool(requests: tuple[EmailRequest] | tuple[SMSRequest]) -> int:
            return requests[0].retries

        requests = [item]
    else:

        @function_tool
        async def tool(requests: tuple[EmailRequest, ...] | tuple[SMSRequest, ...]) -> int:
            return requests[0].retries

        requests = [item]

    payload = {"requests": requests}
    original = copy.deepcopy(payload)
    if shape in ("list", "nested_list", "variadic_tuple"):
        # This is a valid strict-mode response, including the null default sentinel.
        validate(payload, build_strict_openai_schema(tool)["function"]["parameters"])

    args, kwargs = prepare_function_arguments(fnc=tool, json_arguments=payload)

    assert await tool(*args, **kwargs) == retries
    parsed = args[0][0][0] if shape == "nested_list" else args[0][0]
    assert isinstance(parsed, EmailRequest if kind == "email" else SMSRequest)
    assert parsed.note is None  # a genuinely nullable field must keep its null
    assert payload == original


async def test_container_union_keeps_explicit_values_and_omitted_defaults():
    @function_tool
    async def tool(requests: list[EmailRequest] | list[SMSRequest]) -> list[int]:
        return [request.retries for request in requests]

    args, kwargs = prepare_function_arguments(
        fnc=tool,
        json_arguments={"requests": [{"kind": "sms", "retries": 9}, {"kind": "sms"}]},
    )

    assert await tool(*args, **kwargs) == [9, 5]


async def test_container_union_keeps_default_resolution_for_coercible_inputs():
    class First(BaseModel):
        kind: Literal["first"]
        count: int
        retries: int = 2

    class Second(BaseModel):
        kind: Literal["second"]
        count: int
        retries: int = 5

    @function_tool
    async def tool(requests: list[First] | list[Second]) -> tuple[int, int]:
        return requests[0].count, requests[0].retries

    # Non-strict providers can send a string for an integer. No item schema
    # matches it strictly, but default injection must still allow pydantic's
    # existing coercion of the first variant to work.
    args, kwargs = prepare_function_arguments(
        fnc=tool,
        json_arguments={"requests": [{"kind": "first", "count": "3", "retries": None}]},
    )

    assert await tool(*args, **kwargs) == (3, 2)
