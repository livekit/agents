"""Chat-context hygiene for task handoffs."""

from __future__ import annotations

from livekit.agents import llm


def speech_only(chat_ctx: llm.ChatContext) -> llm.ChatContext:
    """The conversation without tool mechanics, for handing to a sub-task.

    Tool calls in the history are scoped to the agent that made them. A
    sub-task whose schema doesn't include those tools will still see them
    being called and imitate them - smaller models invent similar-sounding
    tool names instead of using the ones they actually have. Hand every
    sub-task the words only; anything that matters from a tool result was
    spoken to the caller and survives in the messages.
    """
    return chat_ctx.copy(exclude_function_call=True, exclude_handoff=True)
