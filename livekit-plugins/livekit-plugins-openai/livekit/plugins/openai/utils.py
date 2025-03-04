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
    return [_build_strict_openai_schema(fnc) for fnc in fnc_ctx]


def to_chat_ctx(chat_ctx: llm.ChatContext, cache_key: Any) -> list[ChatCompletionMessageParam]:
    return [_to_chat_item(msg, cache_key) for msg in chat_ctx.items]


def build_legacy_openai_schema(
    ai_function: llm.AIFunction, *, internally_tagged: bool = False
) -> dict[str, Any]:
    """non-strict mode tool description
    see https://serde.rs/enum-representations.html for the internally tagged representation"""
    fnc = llm.utils.serialize_fnc_item(ai_function, strict=False)

    if internally_tagged:
        return {
            "name": fnc["name"],
            "description": fnc["description"],
            "parameters": fnc["schema"],
            "type": "function",
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": fnc["name"],
                "description": fnc["description"],
                "parameters": fnc["schema"],
            },
        }


def _build_strict_openai_schema(
    ai_function: llm.AIFunction,
) -> dict[str, Any]:
    """strict mode tool description"""
    fnc = llm.utils.serialize_fnc_item(ai_function, strict=True)

    return {
        "type": "function",
        "function": {
            "name": fnc["name"],
            "strict": True,
            "description": fnc["description"],
            "parameters": fnc["schema"],
        },
    }


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
    img = llm.utils.serialize_image(image, cache_key)
    b64_data = base64.b64encode(img["data_bytes"]).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{img['media_type']};base64,{b64_data}",
            "detail": img["inference_detail"],
        },
    }
