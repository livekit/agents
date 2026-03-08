from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import ChatContext
from ..metrics import ModelUsage
from .agent_session import AgentSessionOptions, RecordingOptions
from .events import AgentEvent


@dataclass
class SessionReport:
    recording_options: RecordingOptions
    job_id: str
    room_id: str
    room: str
    options: AgentSessionOptions
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

    def to_dict(self) -> dict:
        events_dict: list[dict] = []

        for event in self.events:
            if event.type == "metrics_collected":
                continue  # metrics are too noisy, Cloud is using the chat_history as the source of thruth

            events_dict.append(event.model_dump())

        return {
            "job_id": self.job_id,
            "room_id": self.room_id,
            "room": self.room,
            "events": events_dict,
            "audio_recording_path": (
                str(self.audio_recording_path.absolute()) if self.audio_recording_path else None
            ),
            "audio_recording_started_at": self.audio_recording_started_at,
            "options": {
                "allow_interruptions": self.options.interruption["enabled"],
                "discard_audio_if_uninterruptible": self.options.interruption[
                    "discard_audio_if_uninterruptible"
                ],
                "min_interruption_duration": self.options.interruption["min_duration"],
                "min_interruption_words": self.options.interruption["min_words"],
                "min_endpointing_delay": self.options.endpointing["min_delay"],
                "max_endpointing_delay": self.options.endpointing["max_delay"],
                "max_tool_steps": self.options.max_tool_steps,
                "user_away_timeout": self.options.user_away_timeout,
                "min_consecutive_speech_delay": self.options.min_consecutive_speech_delay,
                "preemptive_generation": self.options.preemptive_generation,
            },
            "chat_history": self.chat_history.to_dict(exclude_timestamp=False),
            "timestamp": self.timestamp,
            "usage": self._usage_to_dict() if self.model_usage else None,
        }

    def _usage_to_dict(self) -> list[dict] | None:
        if self.model_usage is None:
            return None
        return [summary.model_dump(exclude_defaults=True) for summary in self.model_usage]
