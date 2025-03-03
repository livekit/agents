from __future__ import annotations

import base64
import os
from typing import Any, Optional

import boto3
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm import AIFunction, ChatContext, ImageContent

__all__ = ["_to_fnc_ctx", "_to_chat_ctx", "_get_aws_credentials"]


def _get_aws_credentials(api_key: Optional[str], api_secret: Optional[str], region: Optional[str]):
    region = region or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_DEFAULT_REGION must be set via argument or the AWS_DEFAULT_REGION environment variable."
        )

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
    return credentials.access_key, credentials.secret_key, region


def _to_fnc_ctx(fncs: list[AIFunction]) -> list[dict]:
    tools = []
    for fnc in fncs:
        info = llm.utils.get_function_info(fnc)
        model = llm.utils.function_arguments_to_pydantic_model(fnc)
        json_schema = model.model_json_schema()
        tool_spec = {
            "toolSpec": {
                "name": info.name,
                "description": info.description,
                "inputSchema": {"json": json_schema},
            }
        }
        tools.append(tool_spec)
    return tools


def _build_image(image: ImageContent, cache_key: Any) -> dict:
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
            image._cache[cache_key] = utils.images.encode(image.image, opts)
        return {"image": {"format": "jpeg", "source": {"bytes": image._cache[cache_key]}}}
    raise ValueError(f"Unsupported image type: {type(image.image)}")


def _to_chat_ctx(chat_ctx: ChatContext, cache_key: Any) -> tuple[list[dict], dict | None]:
    messages: list[dict] = []
    system_message: dict | None = None
    current_role: str | None = None
    current_content: list[dict] = []

    for msg in chat_ctx.items:
        if msg.type == "message" and msg.role == "system":
            for content in msg.content:
                if isinstance(content, str):
                    system_message = {"text": content}
            continue

        if msg.type == "message":
            role = "assistant" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "assistant"
        elif msg.type == "function_call_output":
            role = "user"

        # if the effective role changed, finalize the previous turn.
        if role != current_role:
            if current_role is not None:
                messages.append({"role": current_role, "content": current_content})
            current_content = []
            current_role = role

        if msg.type == "message":
            for content in msg.content:
                if isinstance(content, str):
                    current_content.append({"text": content})
                elif isinstance(content, ImageContent):
                    current_content.append(_build_image(content, cache_key))
        elif msg.type == "function_call":
            current_content.append({
                "toolUse": {
                    "toolUseId": msg.call_id,
                    "name": msg.name,
                    "input": msg.arguments,
                }
            })
        elif msg.type == "function_call_output":
            tool_response = {
                "toolResult": {
                    "toolUseId": msg.call_id,
                    "content": [],
                    "status": "success",
                }
            }
            if isinstance(msg.output, dict):
                tool_response["toolResult"]["content"].append({"json": msg.output})
            elif isinstance(msg.output, str):
                tool_response["toolResult"]["content"].append({"text": msg.output})
            current_content.append(tool_response)

    # Finalize the last message if there’s any content left
    if current_role is not None and current_content:
        messages.append({"role": current_role, "content": current_content})

    # Ensure the message list starts with a "user" message
    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": [{"text": "(empty)"}]})

    return messages, system_message
