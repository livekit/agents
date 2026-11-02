"""Import-light export counters and invocation-local export suppression."""

from __future__ import annotations

import contextvars
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

_exporting: contextvars.ContextVar[bool] = contextvars.ContextVar("lk_sdk_exporting", default=False)


def exporting() -> bool:
    return _exporting.get()


@contextmanager
def export_context() -> Iterator[None]:
    token = _exporting.set(True)
    try:
        yield
    finally:
        _exporting.reset(token)


Outcome = Literal["dropped", "failed", "conflicts", "incomplete_cleanup"]

_outcomes: Counter[str] = Counter()
_outcomes_lock = threading.Lock()


def record(outcome: Outcome, count: int = 1) -> None:
    """Count a refused record, a failed export, a configuration conflict or a late cleanup."""
    with _outcomes_lock:
        _outcomes[outcome] += count


def diagnostics() -> dict[str, int]:
    """Return process-wide export drops, failures and cleanup counters without exporting them."""
    with _outcomes_lock:
        return {
            outcome: _outcomes[outcome]
            for outcome in ("dropped", "failed", "conflicts", "incomplete_cleanup")
        }
