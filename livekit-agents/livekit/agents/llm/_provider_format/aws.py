from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm


@dataclass
class AWSFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext, generating_reply: bool, *, cache_key: Any
) -> tuple[list[dict], AWSFormatData]:
    messages: list[dict] = []
    system_messages: list[str] = []
    current_role: str | None = None
    current_content: list[dict] = []

    for msg in chat_ctx.items:
        if msg.type == "message" and msg.role == "system":
            system_messages.append(msg.text_content)
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
                elif isinstance(content, llm.ImageContent):
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
    if generating_reply and (not messages or messages[0]["role"] != "user"):
        messages.insert(0, {"role": "user", "content": [{"text": "(empty)"}]})

    return messages, AWSFormatData(system_messages=system_messages)


def _build_image(image: llm.ImageContent, cache_key: Any) -> dict:
    img = llm.utils.serialize_image(image)
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
