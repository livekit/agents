from __future__ import annotations

import json
import logging
import os
from typing import Any

import aioboto3
from botocore.exceptions import NoCredentialsError

from livekit.agents import llm
from livekit.agents.llm import ChatContext, FunctionTool, ImageContent, utils
from livekit.agents.types import NotGivenOr
from livekit.agents.utils import is_given

__all__ = ["to_fnc_ctx", "to_chat_ctx", "get_aws_async_session"]

DEFAULT_REGION = "us-east-1"


def get_aws_async_session(
    api_key: str | None = None,
    api_secret: str | None = None,
    region: str | None = None,
) -> aioboto3.Session:
    """Get an AWS session with the given credentials and region.

    Args:
        api_key: AWS access key id.
        api_secret: AWS secret access key.
        region: AWS region.

    Returns:
        An AWS session.

    Raises:
        NoCredentialsError: If no valid credentials are found.
    """
    # Validate AWS region first
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    if not region:
        raise ValueError(
            "AWS region must be set using the argument or by setting the AWS_REGION environment variable."
        )

    session_params = {"region_name": region}
    if api_key and api_secret:
        session_params.update(
            {
                "aws_access_key_id": api_key,
                "aws_secret_access_key": api_secret,
            }
        )

    session = aioboto3.Session(**session_params)

    # Validate session by checking if we can get credentials
    try:
        session.get_credentials()
    except NoCredentialsError as e:
        logging.error("Unable to locate AWS credentials")
        raise e

    return session


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
