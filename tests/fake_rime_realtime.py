"""Rime lifecycle test providers that also run on the supported older cores."""

from __future__ import annotations

from livekit.agents import llm
from livekit.plugins.rime._realtime_types import RealtimeSession

from .fake_realtime import (
    FakeRealtimeModel as CoreFakeRealtimeModel,
    FakeRealtimeSession as CoreFakeRealtimeSession,
)


def fake_capabilities(**overrides: bool) -> llm.RealtimeCapabilities:
    defaults = {
        "message_truncation": True,
        "turn_detection": True,
        "user_transcription": True,
        "auto_tool_reply_generation": True,
        "audio_output": True,
        "manual_function_calls": True,
        "mutable_chat_context": True,
        "mutable_instructions": True,
        "mutable_tools": True,
        "per_response_tool_choice": True,
        "supports_say": True,
        "supports_overlapping_speech": False,
    }
    defaults.update(overrides)
    return llm.RealtimeCapabilities(
        **{
            key: value
            for key, value in defaults.items()
            if key in llm.RealtimeCapabilities.__dataclass_fields__
        }
    )


class FakeRealtimeSession(CoreFakeRealtimeSession, RealtimeSession):
    pass


class FakeRealtimeModel(CoreFakeRealtimeModel):
    def __init__(
        self,
        *,
        capabilities: llm.RealtimeCapabilities | None = None,
        label: str = "fake.rime.RealtimeModel",
    ) -> None:
        super().__init__(capabilities=capabilities or fake_capabilities(), label=label)

    def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
        session = FakeRealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        session.update_error = self.bring_up_error
        self.created_sessions.append(session)
        return session
