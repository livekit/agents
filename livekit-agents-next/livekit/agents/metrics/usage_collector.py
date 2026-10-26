from __future__ import annotations

import warnings

from .. import _core
from .._generated.usage_collector import UsageSummary
from .base import AgentMetrics

__all__ = ["UsageCollector", "UsageSummary"]


class UsageCollector:
    """
    .. deprecated::
        Use :class:`ModelUsageCollector` instead.
    """

    def __init__(self) -> None:
        warnings.warn(
            "UsageCollector is deprecated. Use ModelUsageCollector from "
            "metrics.model_usage instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._native = _core.UsageCollector()

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def collect(self, metrics: AgentMetrics) -> None:
        self._native.collect(metrics)

    def get_summary(self) -> UsageSummary:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return UsageSummary(**self._native.summary())
