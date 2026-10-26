from __future__ import annotations

import logging

from .. import _core
from ..log import logger as default_logger
from .base import AgentMetrics


def log_metrics(metrics: AgentMetrics, *, logger: logging.Logger | None = None) -> None:
    """Log ``metrics`` at info level with its fields as ``extra`` (the core's
    ``metrics_log_fields``); VAD and EOT inference metrics are not logged."""
    if (entry := _core.metrics_log_fields(metrics)) is not None:
        message, fields = entry
        (logger or default_logger).info(message, extra=fields)
