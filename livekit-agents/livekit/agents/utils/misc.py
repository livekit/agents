"""
General utility functions for agent development including:
- Timestamp generation
- Unique ID creation
- Type validation helpers

Designed for use in real-time media processing pipelines.
"""

from __future__ import annotations

import time
import uuid
from typing import TypeVar

from typing_extensions import TypeGuard

from ..types import NotGiven, NotGivenOr

_T = TypeVar("_T")


def time_ms() -> int:
    """Get current Unix timestamp in milliseconds.
    
    Returns:
        Milliseconds since epoch (UTC)
        
    Used for:
    - Event timestamping
    - Performance metrics
    - Time-sensitive protocol operations
    """
    return int(time.time() * 1000 + 0.5)  # +0.5 for proper rounding


def shortuuid(prefix: str = "") -> str:
    """Generate a shortened UUID for resource identification.
    
    Args:
        prefix: Optional string prefix for namespacing
        
    Returns:
        12-character hex string (16^12 possible combinations)
        
    Example:
        "audio-1a2b3c4d5e6f"
        
    Note: Not cryptographically secure - use for logging/tracing only
    """
    return prefix + str(uuid.uuid4().hex)[:12]


def is_given(obj: NotGivenOr[_T]) -> TypeGuard[_T]:
    """Type guard to check for non-sentinel values in optional parameters.
    
    Usage:
        def process(data: str | NotGiven) -> None:
            if is_given(data):
                # data is now typed as str
                handle_data(data)
    """
    return not isinstance(obj, NotGiven)
