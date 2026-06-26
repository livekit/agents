from __future__ import annotations

import os
import platform
import re
import time
import uuid
from typing import TypeVar
from urllib.parse import urlparse

from typing_extensions import TypeIs

from ..types import NotGiven, NotGivenOr

_T = TypeVar("_T")


def time_ms() -> int:
    return int(time.time() * 1000 + 0.5)


def shortuuid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4().hex)[:12]


def is_given(obj: NotGivenOr[_T]) -> TypeIs[_T]:
    return not isinstance(obj, NotGiven)


def nodename() -> str:
    return platform.node()


def camel_to_snake_case(name: str) -> str:
    return re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    ).lower()


def is_cloud(url: str) -> bool:
    hostname = urlparse(url).hostname
    if hostname is None:
        return False
    return hostname.endswith(".livekit.cloud") or hostname.endswith(".livekit.run")


def is_dev_mode() -> bool:
    """Return whether the agent is running in development mode.

    True when launched via ``console``, ``dev``.
    Reads the ``LIVEKIT_DEV_MODE`` environment variable.
    """
    return os.getenv("LIVEKIT_DEV_MODE") == "1"


def is_hosted() -> bool:
    """Return whether the agent is hosted on LiveKit Cloud."""
    return os.getenv("LIVEKIT_REMOTE_EOT_URL") is not None


def terminal_link(url: str, *, text: str | None = None, color: str = "36") -> str:
    """Format a URL for terminal output.

    Colors the link with the given ANSI SGR ``color`` code (cyan by default) and
    wraps it in an OSC 8 hyperlink escape so supporting terminals render it as a
    clickable link. Terminals without OSC 8 support simply show the colored text.

    Args:
        url: The target URL.
        text: The visible text. Defaults to ``url``.
        color: ANSI SGR color code applied to the visible text.
    """
    text = text if text is not None else url
    return f"\033]8;;{url}\033\\\033[{color}m{text}\033[0m\033]8;;\033\\"
