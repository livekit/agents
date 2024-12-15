from __future__ import annotations

import base64
import json
import os
from typing import Any, Awaitable, Callable, Optional, Union

from livekit import rtc
from livekit.agents import llm, utils

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]


def get_base_url(base_url: Optional[str]) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


def build_oai_message(msg: llm.ChatMessage, cache_key: Any):
    oai_msg: dict[str, Any] = {"role": msg.role}

    if msg.name:
        oai_msg["name"] = msg.name

    # add content if provided
    if isinstance(msg.content, str):
        oai_msg["content"] = msg.content
    elif isinstance(msg.content, dict):
        oai_msg["content"] = json.dumps(msg.content)
    elif isinstance(msg.content, list):
        oai_content: list[dict[str, Any]] = []
        for cnt in msg.content:
            if isinstance(cnt, str):
                oai_content.append({"type": "text", "text": cnt})
            elif isinstance(cnt, llm.ChatImage):
                oai_content.append(_build_oai_image_content(cnt, cache_key))

        oai_msg["content"] = oai_content

    # make sure to provide when function has been called inside the context
    # (+ raw_arguments)
    if msg.tool_calls is not None:
        tool_calls: list[dict[str, Any]] = []
        oai_msg["tool_calls"] = tool_calls
        for fnc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": fnc.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fnc.function_info.name,
                        "arguments": fnc.raw_arguments,
                    },
                }
            )

    # tool_call_id is set when the message is a response/result to a function call
    # (content is a string in this case)
    if msg.tool_call_id:
        oai_msg["tool_call_id"] = msg.tool_call_id

    return oai_msg


def _build_oai_image_content(image: llm.ChatImage, cache_key: Any):
    if isinstance(image.image, str):  # image url
        return {
            "type": "image_url",
            "image_url": {"url": image.image, "detail": image.inference_detail},
        }
    elif isinstance(image.image, rtc.VideoFrame):  # VideoFrame
        if cache_key not in image._cache:
            # inside our internal implementation, we allow to put extra metadata to
            # each ChatImage (avoid to reencode each time we do a chatcompletion request)
            opts = utils.images.EncodeOptions()
            if image.inference_width and image.inference_height:
                opts.resize_options = utils.images.ResizeOptions(
                    width=image.inference_width,
                    height=image.inference_height,
                    strategy="scale_aspect_fit",
                )

            encoded_data = utils.images.encode(image.image, opts)
            image._cache[cache_key] = base64.b64encode(encoded_data).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image._cache[cache_key]}",
                "detail": image.inference_detail,
            },
        }

    raise ValueError(
        "LiveKit OpenAI Plugin: ChatImage must be an rtc.VideoFrame or a URL"
    )
