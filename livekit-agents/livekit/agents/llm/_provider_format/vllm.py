from __future__ import annotations

from typing import Literal

from livekit.agents import llm

from .openai import to_chat_ctx as openai_to_chat_ctx


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
    *,
    inject_dummy_user_message: bool = True,
    dummy_user_message: str = "",
) -> tuple[list[dict], Literal[None]]:
    messages, _ = openai_to_chat_ctx(chat_ctx, inject_dummy_user_message=inject_dummy_user_message)

    if len(messages) == 1 and messages[0]["role"] == "system":
        messages.append({"role": "user", "content": dummy_user_message})
    if len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "assistant":
        messages.insert(1, {"role": "user", "content": "Hello"})
    collated: list[dict] = []
    for msg in messages:
        if len(collated) > 0 and collated[-1]["role"] == msg["role"]:
            collated[-1]["content"] += " " + msg["content"]
        else:
            collated.append(msg)
    return collated, None
