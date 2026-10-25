"""``livekit.rtc`` as the core's room traits see it."""

from __future__ import annotations

from typing import Any


class Room:
    def __init__(self, room: Any) -> None:
        self._room = room
