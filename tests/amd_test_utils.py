from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from livekit.agents.voice.amd import detector as detector_module


@pytest.fixture(autouse=True)
def detector_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "time",
        SimpleNamespace(monotonic=lambda: asyncio.get_running_loop().time(), time=time.time),
    )
