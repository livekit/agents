from __future__ import annotations

from livekit.agents import _core, llm

_DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE = "<instructions>\n{content}\n</instructions>"


def convert_mid_conversation_instructions(
    chat_ctx: llm.ChatContext,
    *,
    role: llm.ChatRole = "user",
    template: str = _DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
) -> llm.ChatContext:
    """Rewrite every system or developer message but a leading preamble as a ``role`` message
    wrapped in ``template``, so it keeps its position; one without text is dropped."""
    from ..._core_view import from_core_item, to_core_items

    plan = _core.convert_mid_conversation_instructions(
        to_core_items(chat_ctx.items), role=role, template=template
    )
    return llm.ChatContext(
        [chat_ctx.items[i] if new is None else from_core_item(new) for i, new in plan]
    )
