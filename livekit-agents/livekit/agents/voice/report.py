import time
from datetime import datetime
from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field
from .agent_session import AgentSession, AgentSessionOptions
from .io import AudioInput, AudioOutput
from .events import AgentEvent
from .recorder_io import RecorderIO, RecorderAudioInput, RecorderAudioOutput
from ..llm import ChatContext
from ..log import logger


@dataclass
class SessionReport:
    job_id: str
    room_id: str
    room: str
    options: AgentSessionOptions
    audio_recording_path: Path | None
    events: list[AgentEvent]
    chat_history: ChatContext
    timestamp: float = field(default_factory=time.time)

    def to_cloud_data(self) -> dict:
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
            "audio_recording_path": str(self.audio_recording_path.absolute()),
            "options": {
                "allow_interruptions": self.options.allow_interruptions,
                "discard_audio_if_uninterruptible": self.options.discard_audio_if_uninterruptible,
                "min_interruption_duration": self.options.min_interruption_duration,
                "min_interruption_words": self.options.min_interruption_words,
                "min_endpointing_delay": self.options.min_endpointing_delay,
                "max_endpointing_delay": self.options.max_endpointing_delay,
                "max_tool_steps": self.options.max_tool_steps,
                "user_away_timeout": self.options.user_away_timeout,
                "min_consecutive_speech_delay": self.options.min_consecutive_speech_delay,
                "preemptive_generation": self.options.preemptive_generation,
            },
            "chat_history": self.chat_history.to_dict(exclude_timestamp=False),
            "timestamp": self.timestamp,
        }
