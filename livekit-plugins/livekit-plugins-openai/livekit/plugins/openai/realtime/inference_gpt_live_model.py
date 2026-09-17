"""Deprecated hosted GPT-Live imports; use :mod:`livekit.agents.inference`."""

from livekit.agents.inference.realtime.gpt_live import (
    InferenceGPTLiveModel,
    InferenceGPTLiveSession,
    InferenceResponsesDelegationOptions,
)

__all__ = [
    "InferenceGPTLiveModel",
    "InferenceGPTLiveSession",
    "InferenceResponsesDelegationOptions",
]
