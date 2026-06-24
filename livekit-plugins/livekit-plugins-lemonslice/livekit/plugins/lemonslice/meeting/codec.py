"""Wire format for meeting chat messages on the agent relay WebSocket (TEXT frames)."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeetingChatMessage:
    sender: str
    text: str
    to: str | None = None


def deserialize_chat(payload: str) -> MeetingChatMessage | None:
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
