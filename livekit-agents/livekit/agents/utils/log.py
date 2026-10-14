import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def log_exceptions(
    msg: str = "",
    logger: logging.Logger = logging.getLogger(),  # noqa: B008
    extra: dict[str, Any] | None = None,
) -> Callable[[F], F]:
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
                    logger.exception(err, extra=extra)
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
                    logger.exception(err, extra=extra)
                    raise

            return cast(F, fn_logs)

    return deco
