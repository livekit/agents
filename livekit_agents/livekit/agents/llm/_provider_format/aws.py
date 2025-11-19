from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from livekit.agents import llm

from .utils import group_tool_calls


@dataclass
class BedrockFormatData:
    system_messages: list[str] | None


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], BedrockFormatData]:
    messages: list[dict] = []
    system_messages: list[str] = []
    current_role: str | None = None
    current_content: list[dict] = []

    for msg in itertools.chain(*(group.flatten() for group in group_tool_calls(chat_ctx))):
        if msg.type == "message" and msg.role == "system" and (text := msg.text_content):
            system_messages.append(text)
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
                    current_content.append(_build_image(content))
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
            current_content.append(
                {
                    "toolResult": {
                        "toolUseId": msg.call_id,
                        "content": [
                            {"json": msg.output}
                            if isinstance(msg.output, dict)
                            else {"text": msg.output}
                        ],
                        "status": "success",
                    }
                }
            )

    # Finalize the last message if there’s any content left
    if current_role is not None and current_content:
        messages.append({"role": current_role, "content": current_content})

    # Ensure the message list starts with a "user" message
    if inject_dummy_user_message and (not messages or messages[0]["role"] != "user"):
        messages.insert(0, {"role": "user", "content": [{"text": "(empty)"}]})

    return messages, BedrockFormatData(system_messages=system_messages)


def _build_image(image: llm.ImageContent) -> dict:
    cache_key = "serialized_image"
    if cache_key not in image._cache:
        image._cache[cache_key] = llm.utils.serialize_image(image)
    img: llm.utils.SerializedImage = image._cache[cache_key]

    if img.external_url:
        raise ValueError("external_url is not supported by AWS Bedrock.")

    return {
        "image": {
            "format": "jpeg",
            "source": {"bytes": img.data_bytes},
        }
    }
