from __future__ import annotations

from pydantic import TypeAdapter

from .. import _core
from .._generated.usage import (
    AgentSessionUsage,
    EOTModelUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    ModelUsage,
    STTModelUsage,
    TTSModelUsage,
    _BaseModelUsage,
)
from .base import AgentMetrics

__all__ = [
    "AgentSessionUsage",
    "EOTModelUsage",
    "InterruptionModelUsage",
    "LLMModelUsage",
    "ModelUsage",
    "ModelUsageCollector",
    "STTModelUsage",
    "TTSModelUsage",
    "_BaseModelUsage",
]

_MODEL_USAGE: TypeAdapter[ModelUsage] = TypeAdapter(ModelUsage)


class ModelUsageCollector:
    """Collects metrics and aggregates usage per provider/model (``_core.ModelUsageCollector``)."""

    def __init__(self) -> None:
        self._native = _core.ModelUsageCollector()

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def collect(self, metrics: AgentMetrics) -> None:
        self._native.collect(metrics)

    def flatten(self) -> list[ModelUsage]:
        """Returns a list of usage summaries, one per model/provider combination."""
        return [_MODEL_USAGE.validate_python(u) for u in self._native.flatten()]
