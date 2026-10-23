from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from livekit.agents.voice.amd import AMD, detector as detector_module


@pytest.fixture(autouse=True)
def detector_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "time",
        SimpleNamespace(monotonic=lambda: asyncio.get_running_loop().time(), time=time.time),
    )


def next_deadline(detector: AMD) -> float | None:
    """Earliest armed detector deadline on the (patched) monotonic clock."""
    return detector._deadline_at(asyncio.get_running_loop().time())
