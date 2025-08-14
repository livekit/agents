from __future__ import annotations

from typing import Literal

from livekit.agents import llm

from .openai import to_chat_ctx as openai_to_chat_ctx


def to_chat_ctx(
    chat_ctx: llm.ChatContext, *, inject_dummy_user_message: bool = True
) -> tuple[list[dict], Literal[None]]:
    messages, _ = openai_to_chat_ctx(chat_ctx, inject_dummy_user_message=inject_dummy_user_message)

    if inject_dummy_user_message and (not messages or messages[-1]["role"] not in ["user", "tool"]):
        messages.append({"role": "user", "content": ""})
    return messages, None
