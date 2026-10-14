from __future__ import annotations

from typing import Literal, get_args

OpenAIRealtimeModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-mini",
    "openai/gpt-realtime-2.1",
    "openai/gpt-realtime-2.1-mini",
]

GoogleRealtimeModels = Literal[
    "google/gemini-3.8-live",
    "google/gemini-3.8-live-extended-thinking",
    "google/gemini-live",
]

XAIRealtimeModels = Literal[
    "xai/grok-voice",
    "xai/grok-voice-latest",
    "xai/grok-voice-3.0",
]

RealtimeModels = OpenAIRealtimeModels | GoogleRealtimeModels | XAIRealtimeModels

OpenAIDuplexModels = Literal["openai/gpt-live-1"]

DuplexModels = OpenAIDuplexModels

# retired ids the gateway still accepts; they stay realtime so llm="..." keeps working
DEPRECATED_REALTIME_MODELS: dict[str, str] = {
    "openai/gpt-realtime-2": "openai/gpt-realtime-2.1",
    "openai/gpt-realtime-1.5": "openai/gpt-realtime-2.1",
    "xai/grok-voice-think-fast-2.0": "xai/grok-voice-3.0",
}

_REALTIME_MODEL_IDS: frozenset[str] = frozenset(
    model for literal in get_args(RealtimeModels) for model in get_args(literal)
) | frozenset(DEPRECATED_REALTIME_MODELS)
_DUPLEX_MODEL_IDS: frozenset[str] = frozenset(get_args(DuplexModels))


def is_realtime_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a realtime (speech-to-speech) model.

    Only the models listed in ``RealtimeModels`` count: a newly released realtime
    model has to be added there before ``llm="..."`` resolves it to a
    ``RealtimeModel``.
    """
    return model in _REALTIME_MODEL_IDS


def is_duplex_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a full-duplex voice model.

    Only the models listed in ``DuplexModels`` count, as with :func:`is_realtime_model`.
    """
    return model in _DUPLEX_MODEL_IDS
