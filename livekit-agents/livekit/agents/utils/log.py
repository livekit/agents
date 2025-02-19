"""
Provides exception logging decorators for agent components.
Ensures consistent error handling across sync/async code paths.

Features:
- Unified error logging for functions/methods
- Async/sync function support
- Customizable log messages
- Stack trace preservation
"""

import asyncio
import functools
import logging
from typing import Any, Callable


def log_exceptions(
    msg: str = "",  # Custom error message prefix
    logger: logging.Logger = logging.getLogger()  # Logger instance to use
) -> Callable[[Any], Any]:
    """Decorator to log exceptions with context.
    
    Automatically handles both sync and async functions.
    
    Usage:
        @log_exceptions("Failed to process audio")
        async def process_audio():
            ...
            
        @log_exceptions(logger=my_logger)
        def sync_operation():
            ...
    """

    def deco(fn: Callable[[Any], Any]):
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_fn_logs(*args: Any, **kwargs: Any):
                """Wrapped async function with error logging"""
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {async_fn_logs.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return async_fn_logs
        else:

            @functools.wraps(fn)
            def fn_logs(*args: Any, **kwargs: Any):
                """Wrapped sync function with error logging"""
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    err = f"Error in {fn_logs.__name__}"
                    if msg:
                        err += f" – {msg}"
                    logger.exception(err)
                    raise

            return fn_logs

    return deco
