from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, List, Optional

from livekit.agents import llm
from livekit.agents.llm.function_context import AIFunction

from google.genai import types

__all__ = ["to_chat_ctx", "to_fnc_ctx"]


def to_fnc_ctx(fncs: list[llm.AIFunction]) -> List[types.FunctionDeclaration]:
    return [_build_gemini_fnc(fnc) for fnc in fncs]


def to_chat_ctx(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> tuple[list[types.Content], Optional[types.Content]]:
    turns: list[types.Content] = []
    system_instruction: Optional[types.Content] = None
    current_role: Optional[str] = None
    parts: list[types.Part] = []

    for msg in chat_ctx.items:
        if msg.type == "message" and msg.role == "system":
            sys_parts = []
            for content in msg.content:
                if isinstance(content, str):
                    sys_parts.append(types.Part(text=content))
            system_instruction = types.Content(parts=sys_parts)
            continue

        if msg.type == "message":
            role = "model" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "model"
        elif msg.type == "function_call_output":
            role = "user"

        # if the effective role changed, finalize the previous turn.
        if role != current_role:
            if current_role is not None and parts:
                turns.append(types.Content(role=current_role, parts=parts))
            parts = []
            current_role = role

        if msg.type == "message":
            for content in msg.content:
                if isinstance(content, str):
                    parts.append(types.Part(text=content))
                elif isinstance(content, dict):
                    parts.append(types.Part(text=json.dumps(content)))
                elif isinstance(content, llm.ImageContent):
                    parts.append(_to_image_part(content, cache_key))
        elif msg.type == "function_call":
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=msg.name,
                        args=json.loads(msg.arguments),
                    )
                )
            )
        elif msg.type == "function_call_output":
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=msg.name,
                        response={"text": msg.output},
                    )
                )
            )

    if current_role is not None and parts:
        turns.append(types.Content(role=current_role, parts=parts))
    return turns, system_instruction


def _to_image_part(image: llm.ImageContent, cache_key: Any) -> types.Part:
    img = llm.utils.serialize_image(image, cache_key)
    return types.Part.from_bytes(data=img.data_bytes, mime_type=img.media_type)


def _build_gemini_fnc(ai_function: AIFunction) -> types.FunctionDeclaration:
    fnc = llm.utils.build_legacy_openai_schema(ai_function, internally_tagged=True)
    json_schema = GeminiSchemaTransformer(fnc["parameters"]).transform()
    return types.FunctionDeclaration(
        name=fnc["name"],
        description=fnc["description"],
        parameters=json_schema,
    )


# https://github.com/googleapis/python-genai/blob/v1.3.0/google/genai/types.py#L75
TYPE_MAPPING = {
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "array": types.Type.ARRAY,
    "object": types.Type.OBJECT,
}


# Convert JSON schema to Gemini SDK types.Schema object
# https://github.com/googleapis/python-genai/blob/v1.3.0/google/genai/types.py#L809
class GeminiSchemaTransformer:
    """
    A class to transform a JSON schema dictionary into a Gemini SDK types.Schema object.
    """

    def __init__(self, schema: dict):
        self.schema = deepcopy(schema)
        self.defs = self.schema.pop("$defs", {})

    def transform(self) -> types.Schema:
        return self._transform_schema(self.schema, refs_stack=())

    def _transform_schema(self, schema: dict, refs_stack: tuple) -> types.Schema:
        if "$ref" in schema:
            ref_key = schema["$ref"].split("/")[-1]
            if ref_key in refs_stack:
                raise ValueError("Recursive $ref is not supported")
            if ref_key not in self.defs:
                raise ValueError(f"Reference {ref_key} not found in $defs")
            new_refs_stack = refs_stack + (ref_key,)
            return self._transform_schema(self.defs[ref_key], new_refs_stack)

        # Dictionary to collect schema fields
        fields = {}

        # Handle anyOf field
        if "anyOf" in schema:
            fields["any_of"] = [
                self._transform_schema(sub_schema, refs_stack) for sub_schema in schema["anyOf"]
            ]
        # Handle type field
        elif "type" in schema:
            type_field = schema["type"]
            if isinstance(type_field, str):
                type_lower = type_field.lower()
                if type_lower not in TYPE_MAPPING:
                    raise ValueError(f"Unsupported type: {type_field}")
                fields["type"] = TYPE_MAPPING[type_lower]
            elif isinstance(type_field, list):
                non_null_types = [t for t in type_field if t != "null"]
                fields["nullable"] = "null" in type_field
                if len(non_null_types) == 1:
                    # single non-null type with nullable
                    type_lower = non_null_types[0].lower()
                    if type_lower not in TYPE_MAPPING:
                        raise ValueError(f"Unsupported type in union: {type_lower}")
                    fields["type"] = TYPE_MAPPING[type_lower]
                elif non_null_types:
                    # union of multiple types
                    fields["any_of"] = [
                        types.Schema(type=TYPE_MAPPING[t.lower()])
                        for t in non_null_types
                        if t.lower() in TYPE_MAPPING
                    ]
                else:
                    raise ValueError("Union contains only null")
            else:
                raise ValueError("Invalid 'type' field")
        else:
            raise ValueError("Schema must specify a 'type' field or 'anyOf'")

        supported_fields = [
            "description",
            "enum",
            "format",
            "title",
            "default",
            "nullable",
            "min_items",
            "max_items",
            "min_length",
            "max_length",
            "minimum",
            "maximum",
            "pattern",
            "min_properties",
            "max_properties",
        ]
        for field in supported_fields:
            if field in schema:
                fields[field] = schema[field]

        # Handle object-specific fields
        if fields.get("type") == types.Type.OBJECT:
            if "properties" in schema:
                fields["properties"] = {
                    key: self._transform_schema(value, refs_stack)
                    for key, value in schema["properties"].items()
                }
            if "required" in schema:
                fields["required"] = schema["required"]

        # Handle array-specific fields
        elif fields.get("type") == types.Type.ARRAY:
            if "items" in schema:
                fields["items"] = self._transform_schema(schema["items"], refs_stack)

        return types.Schema(**fields)
