"""Qualification must preserve stalls, incomplete media and insufficient observations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scripts.bench import media
from scripts.bench.metrics import distribution

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_media_capture_preserves_an_injected_blocking_native_boundary() -> None:
    async def stalled() -> dict[str, object]:
        time.sleep(0.01)
        return {"workload_complete": True}

    result = await media.measure("injected_stall", media.load_budget(), stalled)
    assert result["loop_lag"]["max_ms"] >= 5
    assert result["callback_wall"]["max_ms"] >= 5
    assert result["qualified"] is False


def test_media_qualification_requires_complete_work_and_active_observations(tmp_path: Path) -> None:
    budget = media.load_budget()
    healthy = distribution([100_000] * 1000)
    assert budget.qualified(healthy, healthy, True, 10)
    assert not budget.qualified(healthy, healthy, False, 10)
    assert not budget.qualified(distribution([100_000]), healthy, True, 10)
    assert not budget.qualified(healthy, healthy, True, 0)
    stalled = distribution([100_000] * 1000 + [10_000_000])
    assert not budget.qualified(stalled, healthy, True, 10)
    assert not budget.qualified(healthy, stalled, True, 10)

    manifest = json.loads(Path(media.__file__).with_name("budgets.json").read_text())
    manifest["interfaces"]["media"]["limits"]["loop_lag_max_ms_exclusive"] = 50.0
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="loop_lag_max_ms_exclusive"):
        media.load_budget(path)
