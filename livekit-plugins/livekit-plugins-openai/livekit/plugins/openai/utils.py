from __future__ import annotations

import base64
import os
from collections import OrderedDict
from collections.abc import Awaitable
from typing import Any, Callable, Optional, Union

from livekit.agents import llm

from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]


def get_base_url(base_url: Optional[str]) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


def to_fnc_ctx(fnc_ctx: list[llm.AIFunction]) -> list[ChatCompletionToolParam]:
    return [llm.utils.build_strict_openai_schema(fnc) for fnc in fnc_ctx]


def to_chat_ctx(chat_ctx: llm.ChatContext, cache_key: Any) -> list[ChatCompletionMessageParam]:
    # group the message and function_calls
    item_groups: dict[str, list[llm.ChatItem]] = OrderedDict()
    for item in chat_ctx.items:
        if (item.type == "message" and item.role == "assistant") or item.type == "function_call":
            group_id = item.id.split("/")[0]
            if group_id not in item_groups:
                item_groups[group_id] = []
            item_groups[group_id].append(item)
        else:
            item_groups[item.id] = [item]

    return [_group_to_chat_item(items, cache_key) for items in item_groups.values()]


def _group_to_chat_item(items: list[llm.ChatItem], cache_key: Any) -> ChatCompletionMessageParam:
    if len(items) == 1:
        return _to_chat_item(items[0], cache_key)
    else:
        msg = {"role": "assistant", "tool_calls": []}
        for item in items:
            if item.type == "message":
                assert item.role == "assistant", "only assistant messages can be grouped"
                assert "content" not in msg, "only one assistant message is allowed in a group"

                msg.update(_to_chat_item(item, cache_key))
            elif item.type == "function_call":
                msg["tool_calls"].append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": item.arguments},
                    }
                )
        return msg


def _to_chat_item(msg: llm.ChatItem, cache_key: Any) -> ChatCompletionMessageParam:
    if msg.type == "message":
        oai_content: list[ChatCompletionContentPartParam] = []
        for content in msg.content:
            if isinstance(content, str):
                oai_content.append({"type": "text", "text": content})
            elif isinstance(content, llm.ImageContent):
                oai_content.append(_to_image_content(content, cache_key))

        return {
            "role": msg.role,  # type: ignore
            "content": oai_content,
        }

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


def _to_image_content(image: llm.ImageContent, cache_key: Any) -> ChatCompletionContentPartParam:
    img = llm.utils.serialize_image(image)
    if cache_key not in image._cache:
        image._cache[cache_key] = img.data_bytes
    b64_data = base64.b64encode(image._cache[cache_key]).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{img.media_type};base64,{b64_data}",
            "detail": img.inference_detail,
        },
    }
