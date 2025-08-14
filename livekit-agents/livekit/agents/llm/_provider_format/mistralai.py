from __future__ import annotations

from typing import Any, Literal

from livekit.agents import llm
from livekit.agents.log import logger

from .utils import group_tool_calls


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
) -> tuple[list[dict], Literal[None]]:
    item_groups = group_tool_calls(chat_ctx)
    messages = []

    for group in item_groups:
        if not group.message and not group.tool_calls and not group.tool_outputs:
            continue

        # Convert message to MistralAI format
        if group.message:
            msg = _to_chat_item(group.message)
            if msg:  # Only add non-None messages
                messages.append(msg)

        # Handle tool calls if needed (MistralAI may support tools in the future)
        # For now, we'll skip tool calls since they're not in the basic implementation

    # Ensure we have at least one user message if inject_dummy_user_message is True
    if inject_dummy_user_message and (not messages or messages[0]["role"] != "user"):
        messages.insert(0, {"role": "user", "content": "(empty)"})

    logger.debug(f"MistralAI messages: {messages}")
    return messages, None


def _to_chat_item(msg: llm.ChatItem) -> dict[str, Any] | None:
    if msg.type == "message":
        content = ""
        for content_item in msg.content:
            if isinstance(content_item, str):
                if content:
                    content += "\n"
                content += content_item

        if not content.strip():
            return None

        return {"role": msg.role, "content": content}

    return None
