"""Wire format for meeting chat messages on the agent relay WebSocket."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeetingChatMessage:
    """Parsed meeting chat message from a relay WebSocket TEXT frame.

    Attributes:
        sender: Display name of the message sender.
        text: Message body text.
        to: Optional recipient display name for direct messages.
    """

    sender: str
    text: str
    to: str | None = None


def deserialize_chat(payload: str) -> MeetingChatMessage | None:
    """Parse a meeting chat JSON payload from the relay WebSocket.

    Args:
        payload: Raw JSON string from a TEXT WebSocket frame.

    Returns:
        Parsed chat message, or None if the payload is invalid or not a chat message.
    """
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or obj.get("type") != "chat":
        return None
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    sender = obj.get("sender")
    to = obj.get("to")
    return MeetingChatMessage(
        sender=sender.strip() if isinstance(sender, str) and sender.strip() else "Someone",
        text=text.strip(),
        to=to.strip() if isinstance(to, str) and to.strip() else None,
    )
