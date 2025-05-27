from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm
from livekit.agents.log import logger

from .utils import group_tool_calls


@dataclass
class GoogleFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], GoogleFormatData]:
    turns: list[dict] = []
    system_messages: list[str] = []
    current_role: str | None = None
    parts: list[dict] = []

    for msg in itertools.chain(*(group.flatten() for group in group_tool_calls(chat_ctx))):
        if msg.type == "message" and msg.role == "system" and (text := msg.text_content):
            system_messages.append(text)
            continue

        if msg.type == "message":
            role = "model" if msg.role == "assistant" else "user"
        elif msg.type == "function_call":
            role = "model"
        elif msg.type == "function_call_output":
            role = "user"

        # if the effective role changed, finalize the previous turn.
        if role != current_role:
            if current_role is not None and parts:
                turns.append({"role": current_role, "parts": parts})
            parts = []
            current_role = role

        if msg.type == "message":
            for content in msg.content:
                if content and isinstance(content, str):
                    parts.append({"text": content})
                elif content and isinstance(content, dict):
                    parts.append({"text": json.dumps(content)})
                elif isinstance(content, llm.ImageContent):
                    parts.append(_to_image_part(content))
        elif msg.type == "function_call":
            parts.append(
                {
                    "function_call": {
                        "id": msg.call_id,
                        "name": msg.name,
                        "args": json.loads(msg.arguments or "{}"),
                    }
                }
            )
        elif msg.type == "function_call_output":
            response = {"output": msg.output} if not msg.is_error else {"error": msg.output}
            parts.append(
                {
                    "function_response": {
                        "id": msg.call_id,
                        "name": msg.name,
                        "response": response,
                    }
                }
            )

    if current_role is not None and parts:
        turns.append({"role": current_role, "parts": parts})

    # Gemini requires the last message to end with user's turn before they can generate
    if inject_dummy_user_message and current_role != "user":
        turns.append({"role": "user", "parts": [{"text": "."}]})

    return turns, GoogleFormatData(system_messages=system_messages)


def _to_image_part(image: llm.ImageContent) -> dict[str, Any]:
    cache_key = "serialized_image"
    if cache_key not in image._cache:
        image._cache[cache_key] = llm.utils.serialize_image(image)
    img: llm.utils.SerializedImage = image._cache[cache_key]

    if img.external_url:
        if img.mime_type:
            mime_type = img.mime_type
        else:
            logger.debug("No media type provided for image, defaulting to image/jpeg.")
            mime_type = "image/jpeg"
        return {"file_data": {"file_uri": img.external_url, "mime_type": mime_type}}

    return {"inline_data": {"data": img.data_bytes, "mime_type": img.mime_type}}
