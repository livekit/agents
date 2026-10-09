"""Compatibility imports for hosted Realtime; use :mod:`livekit.agents.inference`."""

from livekit.agents.inference.realtime.openai import (
    RealtimeModel as InferenceRealtimeModel,
    RealtimeSession as InferenceRealtimeSession,
)

__all__ = ["InferenceRealtimeModel", "InferenceRealtimeSession"]
