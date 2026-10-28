from __future__ import annotations

import functools
import inspect
import logging
import os
import platform
import re
import time
import uuid
from collections.abc import Callable
from typing import Any, TypeVar, cast
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
