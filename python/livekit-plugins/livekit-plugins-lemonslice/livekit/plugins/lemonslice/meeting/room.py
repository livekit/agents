"""Types for external meeting integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JoinMeetingResult:
    """Result of adding an active LemonSlice session to an external meeting.

    Attributes:
        websocket_url: WebSocket URL for mixed meeting audio and chat.
        meeting_bot_id: Identifier for the bot instance in the external meeting.
    """

    websocket_url: str
    meeting_bot_id: str
