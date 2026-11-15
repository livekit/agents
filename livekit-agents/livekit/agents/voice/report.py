from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from livekit.protocol.agent_pb import agent_session as agent_pb

from ..llm import ChatContext
from ..metrics import ModelUsage
from ..version import __version__
from .agent_session import AgentSessionOptions
from .events import AgentEvent, CloseReason

# Maps an ErrorEvent's error type to its key in the report's error counts.
_ERROR_KEYS = {
    "llm_error": "llm",
    "stt_error": "stt",
    "tts_error": "tts",
    "realtime_model_error": "realtime",
    "interruption_detection_error": "interruption",
}


def _close_reason_name(reason: CloseReason) -> str:
    """Name of the livekit.agent.CloseReason proto value for an SDK close reason."""
    # The proto calls task completion CR_COMPLETED; every other reason is CR_<member name>.
    member = "COMPLETED" if reason is CloseReason.TASK_COMPLETED else reason.name
    name = f"CR_{member}"
    return name if name in agent_pb.CloseReason.keys() else "CR_UNKNOWN"


@dataclass
class SessionReport:
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
    sdk_version: str = field(default_factory=lambda: __version__)
    """Version of the agents SDK"""
    agent_name: str = ""
    """Name the agent was dispatched with"""
    tags: list[str] = field(default_factory=list)
    """Tags set on the job's Tagger, sorted"""

    def to_dict(self, *, include_chat_history: bool = True, include_events: bool = True) -> dict:
        """Serialize the report. A ``False`` flag omits that key from the result."""
        report: dict = {
            "job_id": self.job_id,
            "room_id": self.room_id,
            "room": self.room,
        }
        if include_events:
            events_dict: list[dict] = []
            for event in self.events:
                if event.type == "metrics_collected":
                    # metrics are too noisy, Cloud is using the chat_history as the source of thruth
                    continue

                events_dict.append(event.model_dump())
            report["events"] = events_dict

        report["audio_recording_path"] = (
            str(self.audio_recording_path.absolute()) if self.audio_recording_path else None
        )
        report["audio_recording_started_at"] = self.audio_recording_started_at
        report["options"] = {
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
            "preemptive_generation": dict(self.options.preemptive_generation),
            "recording_options": dict(self.options.recording_options),
        }
        if include_chat_history:
            report["chat_history"] = self.chat_history.to_dict(exclude_timestamp=False)
        report["timestamp"] = self.timestamp
        report["usage"] = self._usage_to_dict() if self.model_usage else None
        report["sdk_version"] = self.sdk_version
        report["agent_name"] = self.agent_name
        report["tags"] = list(self.tags)
        report["duration_ms"] = int(self.duration * 1000) if self.duration is not None else None
        report["user_turns"] = self._user_turns()
        report["errors"] = self._error_counts()
        report["close_reason"] = self._last_close_reason_name()
        return report

    def _user_turns(self) -> int:
        user_turns = 0
        for item in self.chat_history.items:
            if item.type == "message" and item.role == "user":
                user_turns += 1
        return user_turns

    def _error_counts(self) -> dict[str, int]:
        # every key is always present, so readers need no defaults
        errors = dict.fromkeys(_ERROR_KEYS.values(), 0)
        for event in self.events:
            if event.type == "error":
                key = _ERROR_KEYS.get(getattr(event.error, "type", ""))
                if key is not None:
                    errors[key] += 1
        return errors

    def _last_close_reason_name(self) -> str | None:
        """Proto enum name of the last close event's reason, or None without a close event."""
        name: str | None = None
        for event in self.events:
            if event.type == "close":
                name = _close_reason_name(event.reason)
        return name

    def _usage_to_dict(self) -> list[dict] | None:
        if self.model_usage is None:
            return None
        # exclude_defaults drops ``type``, which tells the usage kinds apart
        return [
            {"type": summary.type, **summary.model_dump(exclude_defaults=True)}
            for summary in self.model_usage
        ]
