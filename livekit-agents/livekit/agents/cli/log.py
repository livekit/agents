from __future__ import annotations

import json
import logging
import re
import traceback
from collections import OrderedDict
from datetime import date, datetime, time, timezone
from inspect import istraceback
from typing import Any, Dict, Tuple

# skip default LogRecord attributes
# http://docs.python.org/library/logging.html#logrecord-attributes
_RESERVED_ATTRS: Tuple[str, ...] = (
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
)


def _merge_record_extra(
    record: logging.LogRecord,
    target: Dict,
):
    for key, value in record.__dict__.items():
        if key not in _RESERVED_ATTRS and not (
            hasattr(key, "startswith") and key.startswith("_")
        ):
            target[key] = value


def _parse_style(formatter: logging.Formatter) -> list[str]:
    """parse the list of fields required by the style"""
    if isinstance(formatter._style, logging.StringTemplateStyle):
        formatter_style_pattern = re.compile(r"\$\{(.+?)\}", re.IGNORECASE)
    elif isinstance(formatter._style, logging.StrFormatStyle):
        formatter_style_pattern = re.compile(r"\{(.+?)\}", re.IGNORECASE)
    elif isinstance(formatter._style, logging.PercentStyle):
        formatter_style_pattern = re.compile(r"%\((.+?)\)", re.IGNORECASE)
    else:
        raise ValueError("Invalid format: %s" % formatter._fmt)

    if formatter._fmt:
        return formatter_style_pattern.findall(formatter._fmt)
    else:
        return []


class JsonFormatter(logging.Formatter):
    class JsonEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (date, datetime, time)):
                return o.isoformat()
            elif istraceback(o):
                return "".join(traceback.format_tb(o)).strip()
            elif type(o) == Exception or isinstance(o, Exception) or type(o) == type:
                return str(o)

            # extra values are formatted as str() if the encoder raises TypeError
            try:
                return super().default(o)
            except TypeError:
                try:
                    return str(o)
                except Exception:
                    return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._required_fields = _parse_style(self)

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record and serializes to json"""
        message_dict: Dict[str, Any] = {}

        if isinstance(record.msg, dict):
            message_dict = record.msg
            record.message = ""
        else:
            record.message = record.getMessage()

        if "asctime" in self._required_fields:
            record.asctime = self.formatTime(record, self.datefmt)

        if record.exc_info and not message_dict.get("exc_info"):
            message_dict["exc_info"] = self.formatException(record.exc_info)
        if not message_dict.get("exc_info") and record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info and not message_dict.get("stack_info"):
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        log_record: Dict[str, Any] = OrderedDict()

        for field in self._required_fields:
            log_record[field] = record.__dict__.get(field)

        log_record.update(message_dict)
        _merge_record_extra(
            record,
            log_record,
        )

        log_record["timestamp"] = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        )

        return json.dumps(
            log_record,
            cls=JsonFormatter.JsonEncoder,
            ensure_ascii=True,
        )


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._esc_codes = {
            "esc_reset": self._esc(0),
            "esc_red": self._esc(31),
            "esc_green": self._esc(32),
            "esc_yellow": self._esc(33),
            "esc_blue": self._esc(34),
            "esc_purple": self._esc(35),
            "esc_cyan": self._esc(36),
            "esc_bold_red": self._esc(1, 31),
        }

        self._level_colors = {
            "DEBUG": self._esc_codes["esc_cyan"],
            "INFO": self._esc_codes["esc_green"],
            "WARNING": self._esc_codes["esc_yellow"],
            "ERROR": self._esc_codes["esc_red"],
            "CRITICAL": self._esc_codes["esc_bold_red"],
        }

        self._required_fields = _parse_style(self)

    @classmethod
    def _esc(cls, *codes: int) -> str:
        return "\033[" + ";".join(str(code) for code in codes) + "m"

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Formats a log record with colors"""

        extra = {}
        _merge_record_extra(record, extra)

        args = {}
        for field in self._required_fields:
            args[field] = record.__dict__.get(field)

        args["esc_levelcolor"] = self._level_colors.get(record.levelname, "")
        args["extra"] = ""
        args.update(self._esc_codes)

        if extra:
            args["extra"] = json.dumps(
                extra, cls=JsonFormatter.JsonEncoder, ensure_ascii=True
            )

        for field in self._required_fields:
            if field in extra:
                del extra[field]

        msg = self._style._fmt % args
        return msg + self._esc_codes["esc_reset"]


def setup_logging(log_level: str, production: bool = True) -> None:
    handler = logging.StreamHandler()

    if not production:
        # colorful logs for dev (improves readability)
        formatter = ColoredFormatter(
            "%(asctime)s - %(esc_levelcolor)s%(levelname)-4s%(esc_reset)s %(name)s - %(message)s %(extra)s",
        )
        handler.setFormatter(formatter)
    else:
        # production logs (serialized of json)
        formatter = JsonFormatter()
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(log_level)
