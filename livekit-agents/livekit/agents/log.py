from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from types import TracebackType
from typing import TypeAlias

DEV_LEVEL = 23
TRACE_LEVEL = 5
logging.addLevelName(DEV_LEVEL, "DEV")
logging.addLevelName(TRACE_LEVEL, "TRACE")


_ExcInfo: TypeAlias = (
    BaseException
    | bool
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)


class Logger(logging.Logger):
    def trace(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        extra: Mapping[str, object] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(
                TRACE_LEVEL,
                message,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel,
            )

    def dev(
        self,
        message: str,
        *args: object,
        exc_info: _ExcInfo = None,
        extra: Mapping[str, object] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        if self.isEnabledFor(DEV_LEVEL):
            self._log(
                DEV_LEVEL,
                message,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel,
            )


_logger_class = logging.getLoggerClass()
logging.setLoggerClass(Logger)
logger: Logger = logging.getLogger("livekit.agents")  # type: ignore[assignment]
logging.setLoggerClass(_logger_class)


# Set by LiveKit Cloud on deployed agents and inherited by job processes; unset elsewhere.
_deployed_region = os.environ.get("LIVEKIT_REGION_NAME") or None


class _GlobalLogFieldsFilter(logging.Filter):
    # Never overwrites an attribute already on the record, so extra={"region": ...} wins.
    def filter(self, record: logging.LogRecord) -> bool:
        if _deployed_region is not None and not hasattr(record, "region"):
            record.region = _deployed_region

        return True


_global_log_fields_filter = _GlobalLogFieldsFilter()


def _add_global_log_fields(handler: logging.Handler) -> None:
    # A filter, not a record factory: the factory runs before makeRecord applies `extra`,
    # so a user-provided extra={"region": ...} would raise KeyError.
    if _deployed_region is None:
        return

    if _global_log_fields_filter not in handler.filters:
        handler.addFilter(_global_log_fields_filter)
