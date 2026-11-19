import logging
from typing import Any

logger = logging.getLogger("livekit.plugins.gradium")


def log_server_message(msg_data: dict[str, Any]) -> None:
    """Log a `{"type": "log"}` frame at the level the server gave it.

    The server sends these, e.g. a warning about an unknown or deprecated setup option,
    to a session whose setup asks for them with `log_level`.
    """
    level = logging.getLevelName(str(msg_data.get("level", "")).upper())
    if not isinstance(level, int):
        level = logging.WARNING
    logger.log(
        level, "Gradium %s: %s", logging.getLevelName(level).lower(), msg_data.get("message")
    )
