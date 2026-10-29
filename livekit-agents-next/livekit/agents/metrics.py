"""The metrics a session emits and the usage it rolls up per model (generated from the core's
declarations), with the usage collector and the log helper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import _core
from ._generated.metrics import (
    AgentMetrics,
    AvatarMetrics,
    EOTInferenceMetrics,
    EOUMetrics,
    InterruptionMetrics,
    LLMMetrics,
    Metadata,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
    _BaseMetrics,
)
from ._generated.usage import (
    EOTModelUsage,
    InterruptionModelUsage,
    LLMModelUsage,
    ModelUsage,
    STTModelUsage,
    TTSModelUsage,
    _BaseModelUsage,
)
from .log import logger as default_logger

__all__ = [
    "LLMMetrics",
    "AgentMetrics",
    "VADMetrics",
    "EOUMetrics",
    "EOTInferenceMetrics",
    "STTMetrics",
    "TTSMetrics",
    "RealtimeModelMetrics",
    "InterruptionMetrics",
    "LLMModelUsage",
    "TTSModelUsage",
    "STTModelUsage",
    "InterruptionModelUsage",
    "EOTModelUsage",
    "ModelUsage",
    "AgentSessionUsage",
    "ModelUsageCollector",
    "log_metrics",
    "AvatarMetrics",
    "Metadata",
    "_BaseMetrics",
    "_BaseModelUsage",
]


@dataclass
class AgentSessionUsage:
    model_usage: list[ModelUsage]


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
        return self._native.flatten()


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    """Log ``metrics`` at info level with its fields as ``extra`` (the core's
    ``metrics_log_fields``); VAD and EOT inference metrics are not logged."""
    if (entry := _core.metrics_log_fields(metrics)) is not None:
        message, fields = entry
        (logger or default_logger).info(message, extra=fields)
