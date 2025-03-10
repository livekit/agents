import datetime
from typing import List, Optional

import pytest
from google.genai import types
from livekit.plugins.google import utils
from pydantic import BaseModel, Field

#  Gemini Schema Tests


# Test for inlining $ref definitions
async def test_json_def_replaced():
    class Location(BaseModel):
        lat: float
        lng: float = 1.1

    class Locations(BaseModel):
        locations: List[Location]

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
                        "lat": {"title": "Lat", "type": types.Type.NUMBER},
                        "lng": {"default": 1.1, "title": "Lng", "type": types.Type.NUMBER},
                    },
                    "required": ["lat"],
                    "title": "Location",
                    "type": types.Type.OBJECT,
                },
                "title": "Locations",
                "type": types.Type.ARRAY,
            }
        },
        "required": ["locations"],
        "title": "Locations",
        "type": types.Type.OBJECT,
    }
    assert gemini_schema == expected_gemini_schema


# Test for handling anyOf (optional field)
async def test_json_def_replaced_any_of():
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        op_location: Optional[Location] = None

    json_schema = Locations.model_json_schema()
    print(json_schema)

    gemini_schema = utils._GeminiJsonSchema(json_schema).simplify()

    # The anyOf containing the Location ref and {"type": "null"} is merged,
    # so op_location becomes the inlined Location with "nullable": True.
    expected_gemini_schema = {
        "properties": {
            "op_location": {
                "properties": {
                    "lat": {"title": "Lat", "type": types.Type.NUMBER},
                    "lng": {"title": "Lng", "type": types.Type.NUMBER},
                },
                "required": ["lat", "lng"],
                "title": "Location",
                "type": types.Type.OBJECT,
                "nullable": True,
                "default": None,
            }
        },
        "title": "Locations",
        "type": types.Type.OBJECT,
    }
    assert gemini_schema == expected_gemini_schema


# Test for recursive $ref – should raise ValueError
async def test_json_def_recursive():
    class Location(BaseModel):
        lat: float
        lng: float
        nested_locations: List["Location"]

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
        ValueError, match=r"Recursive `\$ref`s in JSON Schema are not supported by Gemini"
    ):
        utils._GeminiJsonSchema(json_schema).simplify()


# Test for preserving format, title and description on string fields
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
            "d": {"format": "date", "title": "D", "type": types.Type.STRING},
            "dt": {"format": "date-time", "title": "Dt", "type": types.Type.STRING},
            "t": {
                "format": "time",
                "title": "T",
                "type": types.Type.STRING,
                "description": "",
            },
            "td": {
                "format": "duration",
                "title": "Td",
                "type": types.Type.STRING,
                "description": "my timedelta",
            },
        },
        "required": ["d", "dt", "t", "td"],
        "title": "FormattedStringFields",
        "type": types.Type.OBJECT,
    }
    assert gemini_schema == expected_gemini_schema
