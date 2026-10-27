from __future__ import annotations

import functools
import inspect
import logging
import os
import platform
import re
import string
import time
import uuid
from collections.abc import Callable
from typing import Any, TypeVar, cast
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


def resolve_env_var(val: NotGivenOr[str], *env_vars: str, default: str = "") -> str:
    """
    Resolve an environment variable from a list of potential sources.

    Args:
        val: The value to resolve.
        *env_vars: The environment variables to check. Order matters, the first non-None value will be returned.
        default: The default value to return if no environment variables are set.

    Returns:
        The resolved environment variable.

    Examples:
    >>> resolve_env_var(
    ...     NOT_GIVEN,
    ...     "ABC_URL",
    ...     default="https://agent-gateway.livekit.cloud/v1",
    ... )
    "https://agent-gateway.livekit.cloud/v1"
    """
    if is_given(val):
        return val
    for env_var in env_vars:
        curr_val = os.getenv(env_var, None)
        if curr_val is not None and curr_val != "":
            return curr_val
    return default


F = TypeVar("F", bound=Callable[..., Any])


def log_exceptions(msg: str = "", logger: logging.Logger = logging.getLogger()) -> Callable[[F], F]:  # noqa: B008
    def deco(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_fn_logs(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return cast(F, async_fn_logs)

        else:

            @functools.wraps(fn)
            def fn_logs(*args: Any, **kwargs: Any) -> Any:
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return cast(F, fn_logs)

    return deco
