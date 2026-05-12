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


def is_using_cloud() -> bool:
    """Return whether LiveKit Cloud inference credentials are available.

    Checks the same env-var fallback chain used by the inference LLM / STT / TTS
    constructors: ``LIVEKIT_INFERENCE_API_KEY`` falling back to ``LIVEKIT_API_KEY``,
    and ``LIVEKIT_INFERENCE_API_SECRET`` falling back to ``LIVEKIT_API_SECRET``.
    """
    api_key = os.getenv("LIVEKIT_INFERENCE_API_KEY") or os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_INFERENCE_API_SECRET") or os.getenv("LIVEKIT_API_SECRET")
    return bool(api_key and api_secret)
