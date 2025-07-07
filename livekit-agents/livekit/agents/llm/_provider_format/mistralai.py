from __future__ import annotations

from typing import Any, Literal

from livekit.agents import llm


def to_chat_ctx(
    chat_ctx: llm.ChatContext,
) -> tuple[list[dict], Literal[None]]:
    if isinstance(chat_ctx, llm.ChatContext):
        messages = chat_ctx.to_dict().get("items", [])
        messages_mistral: list[dict[str, Any]] = []
        for element in messages:
            content_list = element.get("content", [])
            if (
                not content_list
                or not isinstance(content_list[0], str)
                or not content_list[0].strip()
            ):
                continue

            content = content_list[0]
            role = element.get("role")

            if role == "assistant":
                messages_mistral.append({"role": "assistant", "content": content})
            elif role == "user":
                messages_mistral.append({"role": "user", "content": content})
            elif role == "system":
                messages_mistral.append({"role": "system", "content": content})

    return messages_mistral, None
