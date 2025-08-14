from __future__ import annotations

from typing import Any, Literal

from livekit.agents import llm


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], Literal[None]]:
    messages: list[dict[str, Any]] = []

    for item in chat_ctx.items:
        if item.type == "message":
            messages.append({"role": item.role, "content": item.text_content})
        elif item.type == "function_call":
            pass

    if inject_dummy_user_message and (not messages or messages[-1]["role"] not in ["user", "tool"]):
        messages.append({"role": "user", "content": ""})

    return messages, None
