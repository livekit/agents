from __future__ import annotations

import base64
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

import boto3
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.function_context import _is_optional_type

__all__ = ["_build_aws_ctx", "_build_tools", "_get_aws_credentials"]


def _get_aws_credentials(
    api_key: Optional[str], api_secret: Optional[str], region: Optional[str]
):
    region = region or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_DEFAULT_REGION must be set using the argument or by setting the AWS_DEFAULT_REGION environment variable."
        )

    # If API key and secret are provided, create a session with them
    if api_key and api_secret:
        session = boto3.Session(
            aws_access_key_id=api_key,
            aws_secret_access_key=api_secret,
            region_name=region,
        )
    else:
        session = boto3.Session(region_name=region)

    credentials = session.get_credentials()
    if not credentials or not credentials.access_key or not credentials.secret_key:
        raise ValueError("No valid AWS credentials found.")
    return credentials.access_key, credentials.secret_key


JSON_SCHEMA_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _build_parameters(arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    properties: Dict[str, dict] = {}
    required: List[str] = []

    for arg_name, arg_info in arguments.items():
        prop = {}
        if hasattr(arg_info, "description") and arg_info.description:
            prop["description"] = arg_info.description

        _, py_type = _is_optional_type(arg_info.type)
        origin = get_origin(py_type)
        if origin is list:
            item_type = get_args(py_type)[0]
            if item_type not in JSON_SCHEMA_TYPE_MAP:
                raise ValueError(f"Unsupported type: {item_type}")
            prop["type"] = "array"
            prop["items"] = {"type": JSON_SCHEMA_TYPE_MAP[item_type]}

            if hasattr(arg_info, "choices") and arg_info.choices:
                prop["items"]["enum"] = list(arg_info.choices)
        else:
            if py_type not in JSON_SCHEMA_TYPE_MAP:
                raise ValueError(f"Unsupported type: {py_type}")

            prop["type"] = JSON_SCHEMA_TYPE_MAP[py_type]

            if arg_info.choices:
                prop["enum"] = list(arg_info.choices)

        properties[arg_name] = prop

        if arg_info.default is inspect.Parameter.empty:
            required.append(arg_name)

    if properties:
        parameters = {"json": {"type": "object", "properties": properties}}
        if required:
            parameters["json"]["required"] = required

        return parameters

    return None


def _build_tools(fnc_ctx: Any) -> List[dict]:
    tools: List[dict] = []
    for fnc_info in fnc_ctx.ai_functions.values():
        parameters = _build_parameters(fnc_info.arguments)

        func_decl = {
            "toolSpec": {
                "name": fnc_info.name,
                "description": fnc_info.description,
                "inputSchema": parameters
                if parameters
                else {"json": {"type": "object", "properties": {}}},
            }
        }

        tools.append(func_decl)
    return tools


def _build_image(image: llm.ChatImage, cache_key: Any) -> dict:
    if isinstance(image.image, str):
        if image.image.startswith("data:image/jpeg;base64,"):
            base64_data = image.image.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                raise ValueError("Invalid base64 data in image URL") from e

            return {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}
        else:
            return {"image": {"format": "jpeg", "source": {"uri": image.image}}}

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

        return {
            "image": {
                "format": "jpeg",
                "source": {
                    "bytes": image._cache[cache_key].encode("utf-8"),
                },
            }
        }
    raise ValueError(f"Unsupported image type: {type(image.image)}")


def _build_aws_ctx(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> Tuple[List[dict], Optional[dict]]:
    messages: List[dict] = []
    system: Optional[dict] = None
    current_role: Optional[str] = None
    current_content: List[dict] = []

    for msg in chat_ctx.messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system = {"text": msg.content}
            continue

        if msg.role == "assistant":
            role = "assistant"
        else:
            role = "user"

        if role != current_role:
            if current_role is not None and current_content:
                messages.append({"role": current_role, "content": current_content})
            current_role = role
            current_content = []

        if msg.tool_calls:
            for fnc in msg.tool_calls:
                current_content.append({
                    "toolUse": {
                        "toolUseId": fnc.tool_call_id,
                        "name": fnc.function_info.name,
                        "input": fnc.arguments,
                    }
                })

        if msg.role == "tool":
            tool_response = {
                "toolResult": {
                    "toolUseId": msg.tool_call_id,
                    "content": [],
                    "status": "success",
                }
            }
            if isinstance(msg.content, dict):
                tool_response["toolResult"]["content"].append({"json": msg.content})
            elif isinstance(msg.content, str):
                tool_response["toolResult"]["content"].append({"text": msg.content})
            current_content.append(tool_response)
        else:
            if msg.content:
                if isinstance(msg.content, str):
                    current_content.append({"text": msg.content})
                elif isinstance(msg.content, dict):
                    current_content.append({"text": json.dumps(msg.content)})
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, str):
                            current_content.append({"text": item})
                        elif isinstance(item, llm.ChatImage):
                            current_content.append(_build_image(item, cache_key))

    if current_role is not None and current_content:
        messages.append({"role": current_role, "content": current_content})

    return messages, system
