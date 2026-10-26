from __future__ import annotations

from livekit.agents import llm

_DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE = "<instructions>\n{content}\n</instructions>"


def convert_mid_conversation_instructions(
    chat_ctx: llm.ChatContext,
    *,
    role: llm.ChatRole = "user",
    template: str = _DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
) -> llm.ChatContext:
    """Rewrite every system or developer message but a leading preamble as a ``role`` message
    wrapped in ``template``, so it keeps its position; one without text is dropped."""
    preamble_allowed = True
    items: list[llm.ChatItem] = []
    for item in chat_ctx.items:
        if item.type == "message" and item.role in ("system", "developer"):
            if preamble_allowed:
                preamble_allowed = False
                items.append(item)
            elif text := item.raw_text_content:
                items.append(
                    llm.ChatMessage(
                        id=item.id,
                        role=role,
                        content=[template.format(content=text)],
                        created_at=item.created_at,
                    )
                )
            continue
        if item.type in ("message", "function_call", "function_call_output"):
            preamble_allowed = False
        items.append(item)
    return llm.ChatContext(items)
