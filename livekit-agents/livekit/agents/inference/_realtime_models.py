from __future__ import annotations

from typing import Literal, get_args

OpenAIRealtimeModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-mini",
    "openai/gpt-realtime-2.1",
    "openai/gpt-realtime-2.1-mini",
]

OpenAIGPTLiveModels = Literal["openai/gpt-live-1"]

XAIRealtimeModels = Literal[
    "xai/grok-voice",
    "xai/grok-voice-latest",
    "xai/grok-voice-think-fast-3.0",
    "xai/grok-voice-think-fast-2.0",
]

RealtimeModels = OpenAIRealtimeModels | OpenAIGPTLiveModels | XAIRealtimeModels

_REALTIME_MODEL_IDS: frozenset[str] = frozenset(
    model for literal in get_args(RealtimeModels) for model in get_args(literal)
)
_GPT_LIVE_MODEL_IDS: frozenset[str] = frozenset(get_args(OpenAIGPTLiveModels))


def is_realtime_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a realtime (speech-to-speech) model.

    Only the models listed in ``RealtimeModels`` count: a newly released realtime
    model has to be added there before ``llm="..."`` resolves it to a
    ``RealtimeModel``, or to a ``GPTLiveModel`` for ``OpenAIGPTLiveModels``.
    """
    return model in _REALTIME_MODEL_IDS


def is_gpt_live_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a GPT-Live model."""
    return model in _GPT_LIVE_MODEL_IDS
