"""Compatibility imports for hosted Realtime; use :mod:`livekit.agents.inference`."""

from livekit.agents.inference.realtime.openai import (
    InferenceRealtimeModel,
    InferenceRealtimeSession,
)

__all__ = ["InferenceRealtimeModel", "InferenceRealtimeSession"]
