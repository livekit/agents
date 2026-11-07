"""What the caller has actually heard, and whether they've answered since, read from the
session's history in order. Something the caller must hear is identified by a token: the id of
the tool call whose result the agent's next reply speaks, or the id of a line spoken as-is.

Nothing here reads what anyone said: only who spoke, in what order, and whether the agent's
words played to the end."""

from __future__ import annotations

from collections.abc import Sequence

from livekit.agents import llm


def delivery(items: Sequence[llm.ChatItem], token: str) -> llm.ChatMessage | None:
    """The agent's message that carried `token` to the caller, if it's been spoken yet."""
    for i, item in enumerate(items):
        if _spoken(item) and item.id == token:
            return item
        if item.type == "function_call_output" and item.call_id == token:
            return next((later for later in items[i + 1 :] if _spoken(later)), None)
    return None


def heard(items: Sequence[llm.ChatItem], token: str) -> bool:
    """The caller heard it to the end: spoken, and not cut off."""
    message = delivery(items, token)
    return message is not None and not message.interrupted


def answered(items: Sequence[llm.ChatItem], token: str) -> bool:
    """The caller heard it to the end and has spoken since."""
    message = delivery(items, token)
    if message is None or message.interrupted:
        return False
    at = next(i for i, item in enumerate(items) if item is message)
    return any(
        item.type == "message" and item.role == "user" for item in items[at + 1 :]
    )


def caller_spoke_last(items: Sequence[llm.ChatItem]) -> bool:
    """The caller's words are the last thing said: the agent hasn't answered them yet."""
    for item in reversed(items):
        if item.type == "message" and item.role == "user":
            return True
        if _spoken(item):
            return False
    return False


def _spoken(item: llm.ChatItem) -> bool:
    return item.type == "message" and item.role == "assistant"
