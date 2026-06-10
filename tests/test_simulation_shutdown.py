from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livekit.agents.job import JobContext

pytestmark = [pytest.mark.unit]


def test_simulator_disconnect_triggers_shutdown() -> None:
    shutdowns: list[str] = []
    ctx = JobContext.__new__(JobContext)
    ctx._on_shutdown = lambda reason: shutdowns.append(reason)

    ctx._on_simulator_disconnected(MagicMock())
    assert shutdowns == ["simulation completed"]
