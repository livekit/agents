from __future__ import annotations

import base64
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

from pydantic import BaseModel

from livekit.rtc import AudioFrame

from ..llm import ChatChunk, ChatContext, GenerationCreatedEvent, LLMOutputEvent
from ..tts import SynthesizedAudio
from ..types import TimedString
from ..vad import VADEvent, VADEventType
from .agent_session import AgentSessionOptions
from .events import AgentEvent, InternalEvent


@dataclass
class SessionReport:
    enable_recording: bool
    include_internal_events: bool
    job_id: str
    room_id: str
    room: str
    options: AgentSessionOptions
    events: list[AgentEvent]
    internal_events: list[InternalEvent]
    chat_history: ChatContext
    audio_recording_path: Path | None = None
    audio_recording_started_at: float | None = None
    """Timestamp when the audio recording started"""
    duration: float | None = None
    started_at: float | None = None
    """Timestamp when the session started"""
    timestamp: float = field(default_factory=time.time)
    """Timestamp when the session report was created, typically at the end of the session"""

    def to_dict(self) -> dict:
        events_dict: list[dict] = []
        internal_events_dict: list[dict] = []

        for event in self.events:
            if event.type == "metrics_collected":
                continue  # metrics are too noisy, Cloud is using the chat_history as the source of thruth

            events_dict.append(event.model_dump())

        if self.include_internal_events:
            for e in self.internal_events:
                if isinstance(e, BaseModel):
                    internal_events_dict.append(e.model_dump())
                elif isinstance(e, SynthesizedAudio):
                    # coming from TTS
                    data = asdict(e)
                    data["frame"] = self._serialize_audio_frame(e.frame)
                    internal_events_dict.append(data)
                elif isinstance(e, LLMOutputEvent):
                    data = asdict(e)
                    if isinstance(e.data, AudioFrame):
                        data["data"] = self._serialize_audio_frame(e.data)
                    elif isinstance(e.data, str):
                        data["data"] = e.data
                    elif isinstance(e.data, TimedString):
                        data["data"] = e.data.to_dict()
                    elif isinstance(e.data, ChatChunk):
                        data["data"] = e.data.model_dump(mode="json")
                    internal_events_dict.append(data)
                elif isinstance(e, VADEvent):
                    # skip inference done events, they are too frequent and too noisy
                    if e.type == VADEventType.INFERENCE_DONE:
                        continue
                    # remove audio frames from VAD event
                    data = asdict(e)
                    data["frames"] = []
                    internal_events_dict.append(data)
                    continue
                elif isinstance(e, GenerationCreatedEvent):
                    data = asdict(e)
                    data["message_stream"] = []
                    data["function_stream"] = []
                    internal_events_dict.append(data)
                    continue
                elif is_dataclass(e):
                    internal_events_dict.append(asdict(e))
                else:
                    raise ValueError(f"Unknown internal event type: {type(e)}")

        return {
            "job_id": self.job_id,
            "room_id": self.room_id,
            "room": self.room,
            "events": events_dict,
            "internal_events": internal_events_dict,
            "audio_recording_path": (
                str(self.audio_recording_path.absolute()) if self.audio_recording_path else None
            ),
            "audio_recording_started_at": self.audio_recording_started_at,
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

    @staticmethod
    def _serialize_audio_frame(frame: AudioFrame) -> dict:
        return {
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": frame.samples_per_channel,
            "data": base64.b64encode(frame.data).decode("utf-8"),
        }
