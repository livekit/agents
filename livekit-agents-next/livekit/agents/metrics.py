"""The performance metrics a session's providers emit (generated from the core's declarations),
and the log helper. Usage is ``livekit.agents.Usage`` (usage.md)."""

from __future__ import annotations

import logging

from . import _core
from ._generated import public_metrics as _public
from ._generated.public_metrics import *  # noqa: F403
from .log import logger as default_logger

__all__ = [
    "log_metrics",
]
__all__ += _public.__all__


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    """Log ``metrics`` at info level with its fields as ``extra`` (the core's
    ``metrics_log_fields``); VAD and EOT inference metrics are not logged."""
    if (entry := _core.metrics_log_fields(metrics)) is not None:
        (logger or default_logger).info(entry.message, extra=entry.extra)
