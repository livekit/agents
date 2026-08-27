from __future__ import annotations

import logging
import os
from typing import Any

DEV_LEVEL = 23
TRACE_LEVEL = 5
logging.addLevelName(DEV_LEVEL, "DEV")
logging.addLevelName(TRACE_LEVEL, "TRACE")


class Logger(logging.Logger):
    def trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, message, args, **kwargs)

    def dev(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(DEV_LEVEL):
            self._log(DEV_LEVEL, message, args, **kwargs)


_logger_class = logging.getLoggerClass()
logging.setLoggerClass(Logger)
logger: Logger = logging.getLogger("livekit.agents")  # type: ignore[assignment]
logging.setLoggerClass(_logger_class)


# LiveKit Cloud injects LIVEKIT_REGION_NAME into deployed agents (e.g. "ca-central").
# The variable is inherited by the job and inference processes, so every process reads
# it directly. It is unset outside of LiveKit Cloud, in which case no field is added.
_deployed_region = os.environ.get("LIVEKIT_REGION_NAME") or None


class _GlobalLogFieldsFilter(logging.Filter):
    """Adds the process-wide log fields to every record passing through a handler.

    Attributes already on the record are never overwritten, so an explicit
    ``extra={"region": ...}`` still wins.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if _deployed_region is not None and not hasattr(record, "region"):
            record.region = _deployed_region

        return True


_global_log_fields_filter = _GlobalLogFieldsFilter()


def _add_global_log_fields(handler: logging.Handler) -> None:
    """Attach the global log fields (currently the deployed region) to ``handler``.

    A filter is used instead of a log record factory because the factory runs before
    ``Logger.makeRecord`` applies ``extra``, which would make any user-provided
    ``extra={"region": ...}`` raise a KeyError.

    No-op when there is nothing to add, and safe to call more than once.
    """
    if _deployed_region is None:
        return

    if _global_log_fields_filter not in handler.filters:
        handler.addFilter(_global_log_fields_filter)
