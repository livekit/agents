"""Compatibility imports for hosted Realtime; use :mod:`livekit.agents.inference`."""

from livekit.agents.inference.realtime import (
    RealtimeModel as InferenceRealtimeModel,
    RealtimeSession as InferenceRealtimeSession,
)

__all__ = ["InferenceRealtimeModel", "InferenceRealtimeSession"]
