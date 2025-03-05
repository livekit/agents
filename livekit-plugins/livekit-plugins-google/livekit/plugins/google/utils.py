from __future__ import annotations

import json
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
    return types.FunctionDeclaration(
        name=fnc["name"],
        description=fnc["description"],
        parameters=_convert_to_gemini_schema(fnc["parameters"]),
    )


JSON_SCHEMA_TYPE_MAP: dict[type, types.Type] = {
    str: "STRING",
    int: "INTEGER",
    float: "NUMBER",
    bool: "BOOLEAN",
    dict: "OBJECT",
    list: "ARRAY",
}


def _convert_to_gemini_schema(json_schema: dict) -> types.Schema | None:
    schema = types.Schema()

    if "type" in json_schema:
        schema.type = JSON_SCHEMA_TYPE_MAP.get(json_schema["type"], "OBJECT")

    if "description" in json_schema:
        schema.description = json_schema["description"]

    if "enum" in json_schema:
        schema.enum = json_schema["enum"]
        if json_schema.get("type") == "integer":
            raise ValueError("Integer enum is not supported by this model.")

    if "items" in json_schema and schema.type == "ARRAY":
        schema.items = _convert_to_gemini_schema(json_schema["items"])

    if "properties" in json_schema and schema.type == "OBJECT":
        properties = {}
        for prop_name, prop_schema in json_schema["properties"].items():
            properties[prop_name] = _convert_to_gemini_schema(prop_schema)
        if not properties:
            return None
        schema.properties = properties

    if "required" in json_schema:
        schema.required = json_schema["required"]

    return schema
