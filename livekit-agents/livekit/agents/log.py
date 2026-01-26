import logging
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
