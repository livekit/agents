"""
Asyncio debugging utilities for performance monitoring and diagnostics.

Provides instrumentation to detect slow async callbacks that may block the event loop.
"""

from __future__ import annotations

import asyncio
import time
from asyncio.base_events import _format_handle  # type: ignore
from typing import Any

from ...log import logger


def hook_slow_callbacks(slow_duration: float) -> None:
    """Instrument asyncio to detect and log slow callback executions.
    
    Args:
        slow_duration: Minimum duration in seconds to consider a callback slow
        
    Usage:
        hook_slow_callbacks(0.1)  # Warn on callbacks taking >100ms
    """
    _run = asyncio.events.Handle._run

    def instrumented(self: Any):
        """Wrapped Handle._run with timing instrumentation."""
        start = time.monotonic()
        val = _run(self)
        dt = time.monotonic() - start
        if dt >= slow_duration:
            logger.warning(
                "Running %s took too long: %.2f seconds",
                _format_handle(self),  # type: ignore
                dt,
            )
        return val

    # Monkey-patch the Handle class
    asyncio.events.Handle._run = instrumented  # type: ignore
