from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import ChatContext
from ..metrics import ModelUsage
from ..version import __version__
from .events import AgentEvent


@dataclass
class SessionReport:
    job_id: str
    room_id: str
    room: str
    options: dict[str, Any] | None
    """The resolved session options (``session.options``)."""
    events: list[AgentEvent]
    chat_history: ChatContext
    audio_recording_path: Path | None = None
    audio_recording_started_at: float | None = None
    """Timestamp when the audio recording started"""
    duration: float | None = None
    started_at: float | None = None
    """Timestamp when the session started"""
    timestamp: float = field(default_factory=time.time)
    """Timestamp when the session report was created, typically at the end of the session"""
    model_usage: list[ModelUsage] | None = None
    """Usage summaries for the session, one per model/provider combination"""
    sdk_version: str = field(default_factory=lambda: __version__)
    """Version of the agents SDK"""

    def to_dict(self) -> dict:
        events_dict = [event.model_dump() for event in self.events]

        return {
            "job_id": self.job_id,
            "room_id": self.room_id,
            "room": self.room,
            "events": events_dict,
            "audio_recording_path": (
                str(self.audio_recording_path.absolute()) if self.audio_recording_path else None
            ),
            "audio_recording_started_at": self.audio_recording_started_at,
            "options": self.options,
            "chat_history": self.chat_history.to_dict(exclude_timestamp=False),
            "timestamp": self.timestamp,
            "usage": self._usage_to_dict() if self.model_usage else None,
            "sdk_version": self.sdk_version,
        }

    def _usage_to_dict(self) -> list[dict] | None:
        if self.model_usage is None:
            return None
        return [summary.model_dump(exclude_defaults=True) for summary in self.model_usage]
