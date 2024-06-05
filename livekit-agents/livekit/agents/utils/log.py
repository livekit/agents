import functools
import logging
from typing import Callable


def log_exceptions(msg: str = "", logger: logging.Logger = logging.getLogger()):
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def fn_logs(*args, **kargs):
            try:
                return fn(*args, **kargs)
            except Exception:
                err = f"Error in {fn.__name__}"
                err += f" – {msg}" if msg else ""
                logger.exception(err)

        return fn_logs

    return deco
