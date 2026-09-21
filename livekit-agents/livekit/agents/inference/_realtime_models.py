from __future__ import annotations

from typing import Literal, get_args

OpenAIRealtimeModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-mini",
    "openai/gpt-realtime-1.5",
    "openai/gpt-realtime-2",
    "openai/gpt-realtime-2.1",
    "openai/gpt-realtime-2.1-mini",
]

XAIRealtimeModels = Literal[
    "xai/grok-voice",
    "xai/grok-voice-latest",
    "xai/grok-voice-think-fast-2.0",
]

RealtimeModels = OpenAIRealtimeModels | XAIRealtimeModels

_REALTIME_MODEL_IDS: frozenset[str] = frozenset(
    model for literal in get_args(RealtimeModels) for model in get_args(literal)
)


def is_realtime_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a realtime (speech-to-speech) model.

    Only the models listed in ``RealtimeModels`` count: a newly released realtime
    model has to be added there before ``llm="..."`` resolves it to a
    ``RealtimeModel``.
    """
    return model in _REALTIME_MODEL_IDS
