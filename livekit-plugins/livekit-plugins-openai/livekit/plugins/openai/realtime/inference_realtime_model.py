"""Deprecated hosted Realtime imports; use :mod:`livekit.agents.inference`."""

from livekit.agents.inference.realtime.openai import (
    InferenceRealtimeModel,
    InferenceRealtimeSession,
)

__all__ = ["InferenceRealtimeModel", "InferenceRealtimeSession"]
