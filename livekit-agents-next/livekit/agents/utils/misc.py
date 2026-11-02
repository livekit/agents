from __future__ import annotations

import functools
import inspect
import logging
import os
import platform
import uuid
from collections.abc import Callable, Coroutine
from typing import Protocol, TypeVar, overload

from typing_extensions import ParamSpec, TypeIs

from ..types import NotGiven, NotGivenOr

_T = TypeVar("_T")


def shortuuid(prefix: str = "") -> str:
    return prefix + str(uuid.uuid4().hex)[:12]


def is_given(obj: NotGivenOr[_T]) -> TypeIs[_T]:
    return not isinstance(obj, NotGiven)


def nodename() -> str:
    return platform.node()


def is_dev_mode() -> bool:
    """Return whether the agent is running in development mode.

    True when launched via ``console``, ``dev``.
    Reads the ``LIVEKIT_DEV_MODE`` environment variable.
    """
    return os.getenv("LIVEKIT_DEV_MODE") == "1"


def is_hosted() -> bool:
    """Return whether the agent is hosted on LiveKit Cloud."""
    return os.getenv("LIVEKIT_REMOTE_EOT_URL") is not None


F = TypeVar("F", bound=Callable[..., object])
_P = ParamSpec("_P")
_R = TypeVar("_R")


class _ExceptionDecorator(Protocol):
    def __call__(self, fn: Callable[_P, _R], /) -> Callable[_P, _R]: ...


def log_exceptions(
    msg: str = "",
    logger: logging.Logger = logging.getLogger(),  # noqa: B008
) -> _ExceptionDecorator:
    @overload
    def deco(
        fn: Callable[_P, Coroutine[object, object, _R]],
    ) -> Callable[_P, Coroutine[object, object, _R]]: ...

    @overload
    def deco(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...

    def deco(fn: Callable[..., object]) -> Callable[..., object]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_fn_logs(*args: object, **kwargs: object) -> object:
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return async_fn_logs

        else:

            @functools.wraps(fn)
            def fn_logs(*args: object, **kwargs: object) -> object:
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return fn_logs

    return deco
