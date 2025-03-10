from __future__ import annotations

import base64
import os
from typing import Any, Awaitable, Callable, Optional, Union

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
    return [_to_chat_item(msg, cache_key) for msg in chat_ctx.items]


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
