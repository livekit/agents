from __future__ import annotations

import base64
import inspect
import json
from typing import Any, Dict, List, Optional, get_args, get_origin

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.function_context import _is_optional_type

from google.genai import types

JSON_SCHEMA_TYPE_MAP: dict[type, types.Type] = {
    str: "STRING",
    int: "INTEGER",
    float: "NUMBER",
    bool: "BOOLEAN",
    dict: "OBJECT",
    list: "ARRAY",
}

__all__ = ["_build_gemini_ctx", "_build_tools"]


def _build_parameters(arguments: Dict[str, Any]) -> types.Schema | None:
    properties: Dict[str, types.Schema] = {}
    required: List[str] = []

    for arg_name, arg_info in arguments.items():
        prop = types.Schema()
        if arg_info.description:
            prop.description = arg_info.description

        _, py_type = _is_optional_type(arg_info.type)
        origin = get_origin(py_type)
        if origin is list:
            item_type = get_args(py_type)[0]
            if item_type not in JSON_SCHEMA_TYPE_MAP:
                raise ValueError(f"Unsupported type: {item_type}")
            prop.type = "ARRAY"
            prop.items = types.Schema(type=JSON_SCHEMA_TYPE_MAP[item_type])

            if arg_info.choices:
                prop.items.enum = arg_info.choices
        else:
            if py_type not in JSON_SCHEMA_TYPE_MAP:
                raise ValueError(f"Unsupported type: {py_type}")

            prop.type = JSON_SCHEMA_TYPE_MAP[py_type]

            if arg_info.choices:
                prop.enum = arg_info.choices
                if py_type is int:
                    raise ValueError(
                        f"Parameter '{arg_info.name}' uses integer choices, not supported by this model."
                    )

        properties[arg_name] = prop

        if arg_info.default is inspect.Parameter.empty:
            required.append(arg_name)

    if properties:
        parameters = types.Schema(type="OBJECT", properties=properties)
        if required:
            parameters.required = required

        return parameters

    return None


def _build_tools(fnc_ctx: Any) -> List[types.FunctionDeclaration]:
    function_declarations: List[types.FunctionDeclaration] = []
    for fnc_info in fnc_ctx.ai_functions.values():
        parameters = _build_parameters(fnc_info.arguments)

        func_decl = types.FunctionDeclaration(
            name=fnc_info.name,
            description=fnc_info.description,
            parameters=parameters,
        )

        function_declarations.append(func_decl)
    return function_declarations


def _build_gemini_ctx(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> tuple[list[types.Content], Optional[types.Content]]:
    turns: list[types.Content] = []
    system_instruction: Optional[types.Content] = None
    current_role: Optional[str] = None
    parts: list[types.Part] = []

    for msg in chat_ctx.messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_instruction = types.Content(parts=[types.Part(text=msg.content)])
            continue

        if msg.role == "assistant":
            role = "model"
        elif msg.role == "tool":
            role = "user"
        else:
            role = "user"

        # If role changed, finalize previous parts into a turn
        if role != current_role:
            if current_role is not None and parts:
                turns.append(types.Content(role=current_role, parts=parts))
            current_role = role
            parts = []

        if msg.tool_calls:
            for fnc in msg.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=fnc.tool_call_id,
                            name=fnc.function_info.name,
                            args=fnc.arguments,
                        )
                    )
                )

        if msg.role == "tool":
            if msg.content:
                if isinstance(msg.content, dict):
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=msg.tool_call_id,
                                name=msg.name,
                                response=msg.content,
                            )
                        )
                    )
                elif isinstance(msg.content, str):
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=msg.tool_call_id,
                                name=msg.name,
                                response={"result": msg.content},
                            )
                        )
                    )
        else:
            if msg.content:
                if isinstance(msg.content, str):
                    parts.append(types.Part(text=msg.content))
                elif isinstance(msg.content, dict):
                    parts.append(types.Part(text=json.dumps(msg.content)))
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(types.Part(text=item))
                        elif isinstance(item, llm.ChatImage):
                            parts.append(_build_gemini_image_part(item, cache_key))

    # Finalize last role's parts if any remain
    if current_role is not None and parts:
        turns.append(types.Content(role=current_role, parts=parts))

    return turns, system_instruction


def _build_gemini_image_part(image: llm.ChatImage, cache_key: Any) -> types.Part:
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
            encoded_data = utils.images.encode(image.image, opts)
            image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")

        return types.Part.from_bytes(
            data=image._cache[cache_key], mime_type="image/jpeg"
        )
    raise ValueError(f"Unsupported image type: {type(image.image)}")
