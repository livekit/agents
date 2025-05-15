from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm
from livekit.agents.log import logger


@dataclass
class GoogleFormatData:
    system_instruction: str | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext, generating_reply: bool, *, cache_key: Any
) -> tuple[list[dict], GoogleFormatData]:
    turns: list[dict] = []
    system_instruction: str | None = None
    current_role: str | None = None
    parts: list[dict] = []

    for msg in chat_ctx.items:
        if msg.type == "message" and msg.role == "system":
            system_instruction = msg.text_content
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
                    parts.append(_to_image_part(content, cache_key))
        elif msg.type == "function_call":
            parts.append(
                {
                    "function_call": {
                        "id": msg.call_id,
                        "name": msg.name,
                        "args": json.loads(msg.arguments),
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
    if generating_reply and current_role != "user":
        turns.append({"role": "user", "parts": [{"text": "."}]})

    return turns, GoogleFormatData(system_instruction=system_instruction)


def _to_image_part(image: llm.ImageContent, cache_key: Any) -> dict[str, Any]:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        if img.mime_type:
            mime_type = img.mime_type
        else:
            logger.debug("No media type provided for image, defaulting to image/jpeg.")
            mime_type = "image/jpeg"
        return {"file_data": {"file_uri": img.external_url, "mime_type": mime_type}}
    if cache_key not in image._cache:
        image._cache[cache_key] = img.data_bytes
    return {"inline_data": {"data": image._cache[cache_key], "mime_type": img.mime_type}}
