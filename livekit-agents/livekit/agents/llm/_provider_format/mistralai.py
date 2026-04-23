from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from livekit.agents import llm

from .utils import group_tool_calls


@dataclass
class MistralFormatData:
    instructions: str | None


def to_conversations_ctx(
    chat_ctx: llm.ChatContext,
) -> tuple[list[dict], MistralFormatData]:
    """Convert ChatContext to Mistral Conversations API entry format.

    Returns:
        A tuple of (entries, instructions) where instructions is the extracted
        system/developer message content (or None if absent).
    """
    item_groups = group_tool_calls(chat_ctx)
    entries: list[dict[str, Any]] = []
    instructions: str | None = None

    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        if group.message:
            item = group.message
            if isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"):
                text_parts = [c for c in item.content if isinstance(c, str)]
                instructions = "\n".join(text_parts) if text_parts else None
                continue

            entry = _to_entry(item)
            if entry is not None:
                entries.append(entry)

        for tool_call in group.tool_calls:
            entries.append(
                {
                    "type": "function.call",
                    "tool_call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
            )

        for tool_output in group.tool_outputs:
            entries.append(
                {
                    "type": "function.result",
                    "tool_call_id": tool_output.call_id,
                    "result": tool_output.output,
                }
            )

    return entries, MistralFormatData(instructions=instructions)


def _to_entry(item: llm.ChatItem) -> dict[str, Any] | None:
    if not isinstance(item, llm.ChatMessage):
        return None

    content = _build_content(item)

    if item.role == "user":
        return {"type": "message.input", "role": "user", "content": content}
    elif item.role == "assistant":
        return {"type": "message.output", "role": "assistant", "content": content}

    return None


def _build_content(msg: llm.ChatMessage) -> str | list[dict[str, Any]]:
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
        return text_content

    if text_content:
        list_content.append({"type": "text", "text": text_content})

    return list_content


def _to_image_content(image: llm.ImageContent) -> dict[str, Any]:
    img = llm.utils.serialize_image(image)
    if img.external_url:
        return {"type": "image_url", "image_url": img.external_url}

    assert img.data_bytes is not None
    b64_data = base64.b64encode(img.data_bytes).decode("utf-8")
    return {"type": "image_url", "image_url": f"data:{img.mime_type};base64,{b64_data}"}
