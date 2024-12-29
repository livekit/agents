import asyncio
import functools
import logging
from typing import Any, Callable

from ..log import FieldsLogger


def log_exceptions(
    msg: str = "", logger: logging.Logger | FieldsLogger = logging.getLogger()
) -> Callable[[Any], Any]:
    def deco(fn: Callable[[Any], Any]):
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_fn_logs(*args: Any, **kwargs: Any):
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
            def fn_logs(*args: Any, **kwargs: Any):
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
