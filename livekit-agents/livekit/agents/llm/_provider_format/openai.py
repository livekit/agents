from __future__ import annotations

import base64
from typing import Any, Literal

from livekit.agents import llm

from .utils import group_tool_calls


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], Literal[None]]:
    item_groups = group_tool_calls(chat_ctx)
    messages = []
    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        # one message can contain zero or more tool calls
        msg = _to_chat_item(group.message) if group.message else {"role": "assistant"}
        tool_calls = [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {"name": tool_call.name, "arguments": tool_call.arguments},
            }
            for tool_call in group.tool_calls
        ]
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

        # append tool outputs following the tool calls
        for tool_output in group.tool_outputs:
            messages.append(_to_chat_item(tool_output))

    return messages, None


def _to_chat_item(msg: llm.ChatItem) -> dict[str, Any]:
    if msg.type == "message":
        list_content: list[dict[str, Any]] = []
        text_content = ""
        for content in msg.content:
            if isinstance(content, str):
                if text_content:
                    text_content += "\n"
                text_content += content
            elif isinstance(content, llm.ImageContent):
                list_content.append(_to_image_content(content))

        if not list_content:
            # certain providers require text-only content in a string vs a list.
            # for max-compatibility, we will combine all text content into a single string.
            return {"role": msg.role, "content": text_content}

        if text_content:
            list_content.append({"type": "text", "text": text_content})

        return {"role": msg.role, "content": list_content}

    elif msg.type == "function_call":
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": msg.call_id,
                    "type": "function",
                    "function": {
                        "name": msg.name,
                        "arguments": msg.arguments,
                    },
                }
            ],
        }

    elif msg.type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": msg.call_id,
            "content": msg.output,
        }


def _to_image_content(image: llm.ImageContent) -> dict[str, Any]:
    cache_key = "serialized_image"  # TODO(long): use hash of encoding options if available
    if cache_key not in image._cache:
        image._cache[cache_key] = llm.utils.serialize_image(image)
    img: llm.utils.SerializedImage = image._cache[cache_key]

    if img.external_url:
        return {
            "type": "image_url",
            "image_url": {
                "url": img.external_url,
                "detail": img.inference_detail,
            },
        }
    assert img.data_bytes is not None
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{img.mime_type};base64,{b64_data}",
            "detail": img.inference_detail,
        },
    }
