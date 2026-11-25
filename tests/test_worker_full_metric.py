"""Unit test for the worker full-capacity counter (`lk_agents_worker_full_total`).

The worker's instantaneous load gauge (`lk_agents_worker_load`) can miss a
transient full-capacity spike that lasts less than one scrape/export interval.
`worker_became_full()` records each transition as a monotonic counter so the
event is never lost to sampling. This test asserts the counter increments per
call and is labelled by node name.
"""

from __future__ import annotations

import pytest

from livekit.agents import utils
from livekit.agents.telemetry import metrics

pytestmark = pytest.mark.unit


def _full_count() -> float:
    value = metrics.WORKER_FULL_COUNTER.labels(nodename=utils.nodename())._value.get()
    return float(value)


def test_worker_became_full_increments_counter() -> None:
    before = _full_count()

    metrics.worker_became_full()
    metrics.worker_became_full()

    assert _full_count() == before + 2
