from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm


@dataclass
class AnthropicFormatData:
    system_instruction: str | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    generating_reply: bool,
    *,
    cache_key: Any,
    cache_control: dict[str, Any] | None,
) -> tuple[list[dict], AnthropicFormatData]:
    messages: list[dict[str, Any]] = []
    system_message: str | None = None
    current_role: str | None = None
    content: list[dict[str, Any]] = []

    for i, msg in enumerate(chat_ctx.items):
        if msg.type == "message" and msg.role == "system":
            system_message = msg.text_content
            continue

        cache_ctrl_i = cache_control if (i == len(chat_ctx.items) - 1) else None
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
                    content.append({"text": c, "type": "text", "cache_control": cache_ctrl_i})
                elif isinstance(c, llm.ImageContent):
                    content.append(_to_image_content(c, cache_key, cache_ctrl=cache_ctrl_i))
        elif msg.type == "function_call":
            content.append(
                {
                    "id": msg.call_id,
                    "type": "tool_use",
                    "name": msg.name,
                    "input": json.loads(msg.arguments or "{}"),
                    "cache_control": cache_ctrl_i,
                }
            )
        elif msg.type == "function_call_output":
            content.append(
                {
                    "tool_use_id": msg.call_id,
                    "type": "tool_result",
                    "content": msg.output,
                    "cache_control": cache_ctrl_i,
                }
            )

    if current_role is not None and content:
        messages.append({"role": current_role, "content": content})

    # ensure the messages starts with a "user" message
    if generating_reply and (not messages or messages[0]["role"] != "user"):
        messages.insert(
            0,
            {
                "role": "user",
                "content": [{"text": "(empty)", "type": "text"}],
            },
        )

    return messages, AnthropicFormatData(system_instruction=system_message)


def _to_image_content(
    image: llm.ImageContent, cache_key: Any, cache_ctrl: dict[str, Any] | None
) -> dict[str, Any]:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        return {
            "type": "image",
            "source": {"type": "url", "url": img.external_url},
            "cache_control": cache_ctrl,
        }
    if cache_key not in image._cache:
        image._cache[cache_key] = img.data_bytes
    b64_data = base64.b64encode(image._cache[cache_key]).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "data": f"data:{img.mime_type};base64,{b64_data}",
            "media_type": img.mime_type,
        },
        "cache_control": cache_ctrl,
    }
