from __future__ import annotations

import base64
import json
from typing import Any, List, Optional

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm import utils
from livekit.agents.llm.function_context import AIFunction

from google.genai import types

JSON_SCHEMA_TYPE_MAP: dict[type, types.Type] = {
    str: "STRING",
    int: "INTEGER",
    float: "NUMBER",
    bool: "BOOLEAN",
    dict: "OBJECT",
    list: "ARRAY",
}

__all__ = ["to_chat_ctx", "to_fnc_ctx"]


def build_gemini_schema(ai_function: AIFunction) -> types.FunctionDeclaration:
    info = llm.utils.get_function_info(ai_function)
    model = llm.utils.function_arguments_to_pydantic_model(ai_function)
    json_schema = model.model_json_schema()
    parameters_schema = convert_to_gemini_schema(json_schema)

    return types.FunctionDeclaration(
        name=info.name,
        description=info.description,
        parameters=parameters_schema,
    )


def convert_to_gemini_schema(json_schema: dict) -> types.Schema | None:
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
        schema.items = convert_to_gemini_schema(json_schema["items"])

    if "properties" in json_schema and schema.type == "OBJECT":
        properties = {}
        for prop_name, prop_schema in json_schema["properties"].items():
            properties[prop_name] = convert_to_gemini_schema(prop_schema)
        if not properties:
            return None
        schema.properties = properties

    if "required" in json_schema:
        schema.required = json_schema["required"]

    return schema


def to_fnc_ctx(fncs: list[llm.AIFunction]) -> List[types.FunctionDeclaration]:
    return [build_gemini_schema(fnc) for fnc in fncs]


def to_chat_ctx(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> tuple[list[types.Content], Optional[types.Content]]:
    turns: list[types.Content] = []
    system_instruction: Optional[types.Content] = None
    current_role: Optional[str] = None
    parts: list[types.Part] = []

    def finalize_turn() -> None:
        nonlocal parts, current_role
        if current_role is not None and parts:
            turns.append(types.Content(role=current_role, parts=parts))
        parts = []

    for msg in chat_ctx.items:
        if msg.type == "message" and msg.role == "system":
            sys_parts = []
            for content in msg.content:
                if isinstance(content, str):
                    sys_parts.append(types.Part(text=content))
            system_instruction = types.Content(parts=sys_parts)
            continue

        if msg.type == "message":
            effective_role = "model" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            effective_role = "model"
        elif msg.type == "function_call_output":
            effective_role = "user"

        # if the effective role changed, finalize the previous turn.
        if effective_role != current_role:
            finalize_turn()
            current_role = effective_role

        if msg.type == "message":
            for content in msg.content:
                if isinstance(content, str):
                    parts.append(types.Part(text=content))
                elif isinstance(content, dict):
                    parts.append(types.Part(text=json.dumps(content)))
                elif isinstance(content, llm.ImageContent):
                    parts.append(to_image_part(content, cache_key))
        elif msg.type == "function_call":
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=msg.name,
                        args=msg.arguments,
                    )
                )
            )
        elif msg.type == "function_call_output":
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=msg.name,
                        response=msg.output,
                    )
                )
            )

    # finalize any remaining parts.
    finalize_turn()
    return turns, system_instruction


def to_image_part(image: llm.ImageContent, cache_key: Any) -> types.Part:
    if isinstance(image.image, str):
        # Check if the string is a Data URL
        if image.image.startswith("data:image/jpeg;base64,"):
            # Extract the base64 part after the comma
            base64_data = image.image.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                raise ValueError("Invalid base64 data in image URL") from e

            return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        else:
            # Assume it's a regular URL
            return types.Part.from_uri(file_uri=image.image, mime_type="image/jpeg")

    elif isinstance(image.image, rtc.VideoFrame):
        if cache_key not in image._cache:
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="scale_aspect_fit",
                )
            image._cache[cache_key] = utils.images.encode(image.image, opts)

        return types.Part.from_bytes(data=image._cache[cache_key], mime_type="image/jpeg")
    raise ValueError(f"Unsupported image type: {type(image.image)}")
