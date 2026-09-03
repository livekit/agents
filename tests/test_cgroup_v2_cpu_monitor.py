"""Regression tests for CGroupV2CPUMonitor impossible-sample handling."""

from __future__ import annotations

from unittest.mock import patch

from livekit.agents.utils.hw.cpu import CGroupV2CPUMonitor


def test_discards_negative_usage_delta():
    monitor = CGroupV2CPUMonitor()
    readings = iter([5_000_000_000, 2_000_000_000])  # non-monotonic
    stamps = iter([0.0, 0.5])

    with (
        patch.object(monitor, "_read_cpu_usage", side_effect=lambda: next(readings)),
        patch.object(monitor, "cpu_count", return_value=8.0),
        patch("livekit.agents.utils.hw.cpu.time.sleep", return_value=None),
        patch("livekit.agents.utils.hw.cpu.time.monotonic", side_effect=lambda: next(stamps)),
    ):
        assert monitor.cpu_percent(interval=0.5) == 0.0


def test_clamps_normal_sample_to_unit_interval():
    monitor = CGroupV2CPUMonitor()
    # 0.25 CPU-seconds over 0.5s on 8 CPUs => 0.0625
    readings = iter([0, 250_000])
    stamps = iter([0.0, 0.5])

    with (
        patch.object(monitor, "_read_cpu_usage", side_effect=lambda: next(readings)),
        patch.object(monitor, "cpu_count", return_value=8.0),
        patch("livekit.agents.utils.hw.cpu.time.sleep", return_value=None),
        patch("livekit.agents.utils.hw.cpu.time.monotonic", side_effect=lambda: next(stamps)),
    ):
        assert abs(monitor.cpu_percent(interval=0.5) - 0.0625) < 1e-9
