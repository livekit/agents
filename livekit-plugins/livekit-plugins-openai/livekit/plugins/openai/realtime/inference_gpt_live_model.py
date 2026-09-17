"""Compatibility imports for hosted GPT-Live; use :mod:`livekit.agents.inference`."""

from livekit.agents.inference.realtime.gpt_live import (
    GPTLiveModel as InferenceGPTLiveModel,
    GPTLiveResponsesDelegationOptions as InferenceResponsesDelegationOptions,
    GPTLiveSession as InferenceGPTLiveSession,
)

__all__ = [
    "InferenceGPTLiveModel",
    "InferenceGPTLiveSession",
    "InferenceResponsesDelegationOptions",
]
