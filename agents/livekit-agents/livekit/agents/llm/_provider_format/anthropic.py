from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm

from .utils import group_tool_calls


@dataclass
class AnthropicFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], AnthropicFormatData]:
    messages: list[dict[str, Any]] = []
    system_messages: list[str] = []
    current_role: str | None = None
    content: list[dict[str, Any]] = []

    chat_items: list[llm.ChatItem] = []
    for group in group_tool_calls(chat_ctx):
        chat_items.extend(group.flatten())

    for msg in chat_items:
        if msg.type == "message" and msg.role == "system" and (text := msg.text_content):
            system_messages.append(text)
            continue

        if msg.type == "message":
            role = "assistant" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "assistant"
        elif msg.type == "function_call_output":
            role = "user"

        if role != current_role:
            if current_role is not None and content:
                messages.append({"role": current_role, "content": content})
            content = []
            current_role = role

        if msg.type == "message":
            for c in msg.content:
                if c and isinstance(c, str):
                    content.append({"text": c, "type": "text"})
                elif isinstance(c, llm.ImageContent):
                    content.append(_to_image_content(c))
        elif msg.type == "function_call":
            content.append(
                {
                    "id": msg.call_id,
                    "type": "tool_use",
                    "name": msg.name,
                    "input": json.loads(msg.arguments or "{}"),
                }
            )
        elif msg.type == "function_call_output":
            content.append(
                {
                    "tool_use_id": msg.call_id,
                    "type": "tool_result",
                    "content": msg.output,
                    "is_error": msg.is_error,
                }
            )

    if current_role is not None and content:
        messages.append({"role": current_role, "content": content})

    # ensure the messages starts with a "user" message
    if inject_dummy_user_message and (not messages or messages[0]["role"] != "user"):
        messages.insert(
            0,
            {
                "role": "user",
                "content": [{"text": "(empty)", "type": "text"}],
            },
        )

    return messages, AnthropicFormatData(system_messages=system_messages)


def _to_image_content(image: llm.ImageContent) -> dict[str, Any]:
    cache_key = "serialized_image"
    if cache_key not in image._cache:
        image._cache[cache_key] = llm.utils.serialize_image(image)
    img: llm.utils.SerializedImage = image._cache[cache_key]

    if img.external_url:
        return {
            "type": "image",
            "source": {"type": "url", "url": img.external_url},
        }

    assert img.data_bytes is not None
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "data": f"data:{img.mime_type};base64,{b64_data}",
            "media_type": img.mime_type,
        },
    }
