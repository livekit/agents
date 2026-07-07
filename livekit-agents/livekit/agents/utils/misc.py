from __future__ import annotations

import os
import platform
import re
import string
import time
import uuid
from typing import Any, TypeVar
from urllib.parse import urlparse

from typing_extensions import TypeIs

from ..log import logger
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


def safe_render(template: str, data: dict[str, object]) -> str:
    """Fill *template* placeholders from *data*.

    Missing keys log a warning and become empty strings.
    ``None`` values also become empty strings.
    Nested dicts are converted to namespaces for dotted access
    (e.g. ``{audio_recognition.stt_context.emotion}``).
    """
    from types import SimpleNamespace

    def _to_ns(v: object) -> object:
        if isinstance(v, dict):
            return SimpleNamespace(**{k: _to_ns(val) for k, val in v.items()})
        return v

    def _collect_paths(d: dict[str, object], prefix: str = "") -> list[str]:
        paths: list[str] = []
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                paths.extend(_collect_paths(v, full))
            else:
                paths.append(full)
        return paths

    resolved = {k: _to_ns(v) for k, v in data.items()}
    available_paths = _collect_paths(data)

    class _Fmt(string.Formatter):
        def get_field(self, field_name: str, args: Any, kwargs: Any) -> Any:
            try:
                return super().get_field(field_name, args, kwargs)
            except (KeyError, AttributeError, TypeError):
                logger.error(
                    "template placeholder '{%s}' has no value (available: %s)",
                    field_name,
                    available_paths,
                )
                return ("", field_name)

        def format_field(self, value: Any, format_spec: str) -> str:
            if value is None:
                return ""
            return str(super().format_field(value, format_spec))

    return _Fmt().vformat(template, (), resolved)
