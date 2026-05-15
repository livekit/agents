import logging

from livekit.agents.log import Logger

# Use the agents `Logger` subclass so `logger.trace(...)` works in this
# plugin alongside the standard levels. Same setLoggerClass dance as
# `livekit.agents.log` — temporarily install, instantiate, restore.
_logger_class = logging.getLoggerClass()
logging.setLoggerClass(Logger)
logger: Logger = logging.getLogger("livekit.plugins.turn_detector")  # type: ignore[assignment]
logging.setLoggerClass(_logger_class)
