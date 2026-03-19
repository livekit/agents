from __future__ import annotations

import logging
import os
import sys

DEV_LEVEL = 23
logging.addLevelName(DEV_LEVEL, "DEV")

logger = logging.getLogger("livekit.agents")
logger.addHandler(logging.NullHandler())

_LOG_JSON_ENV = "LIVEKIT_AGENTS_LOG_JSON"


def configure_logging(
    *,
    level: int | str = logging.DEBUG,
    handler: logging.Handler | None = None,
    formatter: logging.Formatter | None = None,
    json: bool = False,
) -> None:
    """Configure logging for livekit-agents.

    Call this before your agent entrypoint to customise log output.
    When ``json=True`` the built-in :class:`JsonFormatter` is used regardless
    of *formatter*.

    The ``json`` setting is automatically inherited by child processes
    (e.g. the hot-reload subprocess spawned by ``dev``).
    """
    from .cli.log import JsonFormatter

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    if json:
        handler.setFormatter(JsonFormatter())
        os.environ[_LOG_JSON_ENV] = "1"
    elif formatter is not None:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

    # ensure the livekit.agents logger isn't blocked by a parent (e.g. "livekit")
    # that was set to a higher level
    if logger.level == logging.NOTSET:
        logger.setLevel(level)
