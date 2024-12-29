import logging
from typing import Any

DEV_LEVEL = 23
logging.addLevelName(DEV_LEVEL, "DEV")


class FieldsLogger(logging.LoggerAdapter):
    """A logger adapter that adds fields to each log message"""

    def __init__(self, logger: logging.Logger, fields: dict[str, Any] = {}) -> None:
        super().__init__(logger, fields)

    def process(self, msg, kwargs):
        # Merge global fields with any per-message extras
        extras = self.extra
        if "extra" in kwargs:
            extras = self.extra.copy()
            extras.update(kwargs["extra"])
        kwargs["extra"] = extras
        return msg, kwargs

    def with_fields(self, fields: dict[str, Any]) -> "FieldsLogger":
        return FieldsLogger(self.logger, {**self.extra, **fields})


logger = logging.getLogger("livekit.agents")
