import datetime

import pytest
from google.genai import types
from pydantic import BaseModel, Field

from livekit.agents import llm
from livekit.agents.llm import function_tool
from livekit.plugins.google import utils

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

#  Gemini Schema Tests


# Test for inlining $ref definitions
async def test_json_def_replaced():
    class Location(BaseModel):
        lat: float
        lng: float = 1.1

    class Locations(BaseModel):
        locations: list[Location]

    json_schema = Locations.model_json_schema()
    # Original schema with $defs as produced by Pydantic.
    expected_schema = {
        "$defs": {
            "Location": {
                "properties": {
                    "lat": {"title": "Lat", "type": "number"},
                    "lng": {"default": 1.1, "title": "Lng", "type": "number"},
                },
                "required": ["lat"],
                "title": "Location",
                "type": "object",
            }
        },
        "properties": {
            "locations": {
                "items": {"$ref": "#/$defs/Location"},
                "title": "Locations",
                "type": "array",
            }
        },
        "required": ["locations"],
        "title": "Locations",
        "type": "object",
    }
    assert json_schema == expected_schema

    gemini_schema = utils._GeminiJsonSchema(json_schema).simplify()

    expected_gemini_schema = {
        "properties": {
            "locations": {
                "items": {
                    "properties": {
                        "lat": {"type": types.Type.NUMBER},
                        "lng": {"type": types.Type.NUMBER},
                    },
                    "required": ["lat"],
                    "type": types.Type.OBJECT,
                },
                "type": types.Type.ARRAY,
            }
        },
        "required": ["locations"],
        "type": types.Type.OBJECT,
    }
    assert gemini_schema == expected_gemini_schema


# Test for handling anyOf (optional field)
async def test_json_def_replaced_any_of():
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        op_location: Location | None = None

    json_schema = Locations.model_json_schema()

    gemini_schema = utils._GeminiJsonSchema(json_schema).simplify()

    # The anyOf containing the Location ref and {"type": "null"} is merged,
    # so op_location becomes the inlined Location with "nullable": True.
    expected_gemini_schema = {
        "properties": {
            "op_location": {
                "properties": {
                    "lat": {"type": types.Type.NUMBER},
                    "lng": {"type": types.Type.NUMBER},
                },
                "required": ["lat", "lng"],
                "type": types.Type.OBJECT,
                "nullable": True,
            }
        },
        "type": types.Type.OBJECT,
    }
    assert gemini_schema == expected_gemini_schema


# Test for recursive $ref – should raise ValueError
async def test_json_def_recursive():
    class Location(BaseModel):
        lat: float
        lng: float
        nested_locations: list["Location"]

    Location.model_rebuild()
    json_schema = Location.model_json_schema()
    expected_schema = {
        "$defs": {
            "Location": {
                "properties": {
                    "lat": {"title": "Lat", "type": "number"},
                    "lng": {"title": "Lng", "type": "number"},
                    "nested_locations": {
                        "items": {"$ref": "#/$defs/Location"},
                        "title": "Nested Locations",
                        "type": "array",
                    },
                },
                "required": ["lat", "lng", "nested_locations"],
                "title": "Location",
                "type": "object",
            }
        },
        "$ref": "#/$defs/Location",
    }
    assert json_schema == expected_schema

    with pytest.raises(
        ValueError,
        match=r"Recursive `\$ref`s in JSON Schema are not supported by Gemini",
    ):
        utils._GeminiJsonSchema(json_schema).simplify()


# Test for preserving format, and description on string fields
async def test_json_def_date():
    class FormattedStringFields(BaseModel):
        d: datetime.date
        dt: datetime.datetime
        t: datetime.time = Field(description="")
        td: datetime.timedelta = Field(description="my timedelta")

    json_schema = FormattedStringFields.model_json_schema()
    expected_schema = {
        "properties": {
            "d": {"format": "date", "title": "D", "type": "string"},
            "dt": {"format": "date-time", "title": "Dt", "type": "string"},
            "t": {"format": "time", "title": "T", "type": "string", "description": ""},
            "td": {
                "format": "duration",
                "title": "Td",
                "type": "string",
                "description": "my timedelta",
            },
        },
        "required": ["d", "dt", "t", "td"],
        "title": "FormattedStringFields",
        "type": "object",
    }
    assert json_schema == expected_schema

    gemini_schema = utils._GeminiJsonSchema(json_schema).simplify()
    expected_gemini_schema = {
        "properties": {
            "d": {"format": "date", "type": types.Type.STRING},
            "dt": {"format": "date-time", "type": types.Type.STRING},
            "t": {
                "format": "time",
                "type": types.Type.STRING,
                "description": "",
            },
            "td": {
                "format": "duration",
                "type": types.Type.STRING,
                "description": "my timedelta",
            },
        },
        "required": ["d", "dt", "t", "td"],
        "type": types.Type.OBJECT,
    }
    assert gemini_schema == expected_gemini_schema


#  FunctionTool: Gemini FunctionDeclaration tests


@function_tool
async def save_contact(fields: dict[str, str]) -> None:
    """Save the contact fields gathered in the conversation (free-form keys)."""


@function_tool
async def ping() -> None:
    """A tool without parameters."""


# Test for Gemini Text API. It should emit parameters_json_schema
async def test_function_tool_uses_parameters_json_schema_on_text_api():
    [schema] = llm.ToolContext([save_contact]).parse_function_tools("google")

    assert "parameters" not in schema
    fields = schema["parameters_json_schema"]["properties"]["fields"]
    assert fields["type"] == "object"
    assert fields["additionalProperties"] == {"type": "string"}
    assert schema["parameters_json_schema"]["required"] == ["fields"]

    decl = types.FunctionDeclaration.model_validate(schema)
    assert decl.parameters is None
    assert decl.parameters_json_schema is not None


# Test for Gemini Live API. It should keep the simplified legacy parameters
async def test_function_tool_uses_legacy_parameters_on_live_api():
    [schema] = llm.ToolContext([save_contact]).parse_function_tools(
        "google", use_parameters_json_schema=False
    )

    assert "parameters_json_schema" not in schema
    assert schema["parameters"] == {
        "type": types.Type.OBJECT,
        "properties": {"fields": {"type": types.Type.OBJECT}},
        "required": ["fields"],
    }
    types.FunctionDeclaration.model_validate(schema)


# Test for a FunctionTool with no parameters. Both APIs should omit the schema
async def test_function_tool_without_parameters():
    [text_schema] = llm.ToolContext([ping]).parse_function_tools("google")
    assert text_schema["parameters_json_schema"] is None
    assert "parameters" not in text_schema

    [live_schema] = llm.ToolContext([ping]).parse_function_tools(
        "google", use_parameters_json_schema=False
    )
    assert live_schema["parameters"] is None
    assert "parameters_json_schema" not in live_schema


# Test for RawFunctionTool and FunctionTool matching on the text API
async def test_raw_function_tool_matches_function_tool_on_text_api():
    raw = function_tool(
        lambda raw_arguments: None,
        raw_schema={
            "name": "save_contact",
            "description": "Save the contact fields gathered in the conversation (free-form keys).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {"type": "object", "additionalProperties": {"type": "string"}}
                },
                "required": ["fields"],
            },
        },
    )
    [raw_schema] = llm.ToolContext([raw]).parse_function_tools("google")
    [fnc_schema] = llm.ToolContext([save_contact]).parse_function_tools("google")

    raw_fields = raw_schema["parameters_json_schema"]["properties"]["fields"]
    fnc_fields = fnc_schema["parameters_json_schema"]["properties"]["fields"]
    assert fnc_fields["additionalProperties"] == raw_fields["additionalProperties"]
    assert fnc_schema["parameters_json_schema"]["required"] == ["fields"]


# Test for create_tools_config emitting parameters_json_schema
async def test_create_tools_config_emits_parameters_json_schema():
    tools, _ = utils.create_tools_config(llm.ToolContext([save_contact]))

    [tool] = tools
    assert tool.function_declarations is not None
    [decl] = tool.function_declarations
    assert decl.parameters is None
    assert decl.parameters_json_schema is not None
    assert decl.parameters_json_schema["properties"]["fields"]["additionalProperties"] == {
        "type": "string"
    }
