from __future__ import annotations

from typing import Literal, get_args

OpenAIRealtimeModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-2.1",
    "openai/gpt-realtime-2.1-mini",
]

XAIRealtimeModels = Literal[
    "xai/grok-voice-latest",
    "xai/grok-voice-think-fast-2.0",
]

RealtimeModels = OpenAIRealtimeModels | XAIRealtimeModels

_REALTIME_MODEL_IDS: frozenset[str] = frozenset(
    model for literal in get_args(RealtimeModels) for model in get_args(literal)
)


def is_realtime_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a realtime (speech-to-speech) model.

    Models released after this version aren't in ``RealtimeModels`` yet, so a name
    containing ``realtime`` or ``voice`` (e.g. ``openai/gpt-realtime-3``) also counts.
    """
    if model in _REALTIME_MODEL_IDS:
        return True

    name = model.split("/")[-1]
    return "realtime" in name or "voice" in name
