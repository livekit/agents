from __future__ import annotations

import json
from typing import Any

import aioboto3
import boto3
from botocore.exceptions import NoCredentialsError

from livekit.agents import llm
from livekit.agents.llm import ChatContext, FunctionTool, ImageContent, utils

__all__ = ["to_fnc_ctx", "to_chat_ctx", "get_aws_async_session"]
DEFAULT_REGION = "us-east-1"


def get_aws_async_session(
    region: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> aioboto3.Session:
    _validate_aws_credentials(api_key, api_secret)
    session = aioboto3.Session(
        aws_access_key_id=api_key,
        aws_secret_access_key=api_secret,
        region_name=region or DEFAULT_REGION,
    )
    return session


def _validate_aws_credentials(
    api_key: str | None = None,
    api_secret: str | None = None,
) -> None:
    try:
        session = boto3.Session(aws_access_key_id=api_key, aws_secret_access_key=api_secret)
        creds = session.get_credentials()
        if not creds:
            raise ValueError("No credentials found")
    except (NoCredentialsError, Exception) as e:
        raise ValueError(f"Unable to locate valid AWS credentials: {str(e)}") from e


def to_fnc_ctx(fncs: list[FunctionTool]) -> list[dict]:
    return [_build_tool_spec(fnc) for fnc in fncs]


def to_chat_ctx(chat_ctx: ChatContext, cache_key: Any) -> tuple[list[dict], dict | None]:
    messages: list[dict] = []
    system_message: dict | None = None
    current_role: str | None = None
    current_content: list[dict] = []

    for msg in chat_ctx.items:
        if msg.type == "message" and msg.role == "system":
            for content in msg.content:
                if content and isinstance(content, str):
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
            if current_content and current_role is not None:
                messages.append({"role": current_role, "content": current_content})
            current_content = []
            current_role = role

        if msg.type == "message":
            for content in msg.content:
                if content and isinstance(content, str):
                    current_content.append({"text": content})
                elif isinstance(content, ImageContent):
                    current_content.append(_build_image(content, cache_key))
        elif msg.type == "function_call":
            current_content.append(
                {
                    "toolUse": {
                        "toolUseId": msg.call_id,
                        "name": msg.name,
                        "input": json.loads(msg.arguments or "{}"),
                    }
                }
            )
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


def _build_tool_spec(fnc: FunctionTool) -> dict:
    fnc = llm.utils.build_legacy_openai_schema(fnc, internally_tagged=True)
    return {
        "toolSpec": _strip_nones(
            {
                "name": fnc["name"],
                "description": fnc["description"] if fnc["description"] else None,
                "inputSchema": {"json": fnc["parameters"] if fnc["parameters"] else {}},
            }
        )
    }


def _build_image(image: ImageContent, cache_key: Any) -> dict:
    img = utils.serialize_image(image)
    if img.external_url:
        raise ValueError("external_url is not supported by AWS Bedrock.")
    if cache_key not in image._cache:
        image._cache[cache_key] = img.data_bytes
    return {
        "image": {
            "format": "jpeg",
            "source": {"bytes": image._cache[cache_key]},
        }
    }


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
