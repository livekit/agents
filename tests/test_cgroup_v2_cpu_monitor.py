"""Regression tests for CGroupV2CPUMonitor impossible-sample handling."""

from __future__ import annotations

import io
from unittest.mock import mock_open, patch

import pytest

from livekit.agents.utils.hw.cpu import (
    CGroupV2CPUMonitor,
)

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize(
    ("num_cpus", "usage_usec"),
    [(1.0, 550_000), (0.5, 275_000)],
)
def test_keeps_quota_boundary_burst_as_saturation(num_cpus: float, usage_usec: int):
    monitor = CGroupV2CPUMonitor()
    # A sample can cross a quota-period boundary, making it exceed the quota's
    # average rate while remaining below the host's physical capacity.
    readings = iter([0, usage_usec])
    stamps = iter([0.0, 0.5])

    with (
        patch.object(monitor, "_read_cpu_usage", side_effect=lambda: next(readings)),
        patch.object(monitor, "cpu_count", return_value=num_cpus),
        patch("livekit.agents.utils.hw.cpu.psutil.cpu_count", return_value=8),
        patch("livekit.agents.utils.hw.cpu.time.sleep", return_value=None),
        patch("livekit.agents.utils.hw.cpu.time.monotonic", side_effect=lambda: next(stamps)),
    ):
        assert monitor.cpu_percent(interval=0.5) == 1.0


def test_uses_measured_elapsed_time_not_requested_interval():
    monitor = CGroupV2CPUMonitor()
    readings = iter([0, 250_000])
    stamps = iter([0.0, 1.0])

    with (
        patch.object(monitor, "_read_cpu_usage", side_effect=lambda: next(readings)),
        patch.object(monitor, "cpu_count", return_value=1.0),
        patch("livekit.agents.utils.hw.cpu.psutil.cpu_count", return_value=8),
        patch("livekit.agents.utils.hw.cpu.time.sleep", return_value=None),
        patch("livekit.agents.utils.hw.cpu.time.monotonic", side_effect=lambda: next(stamps)),
    ):
        assert monitor.cpu_percent(interval=0.5) == 0.25


def test_discards_usage_above_host_capacity():
    monitor = CGroupV2CPUMonitor()
    readings = iter([0, 4_300_000])
    stamps = iter([0.0, 0.5])

    with (
        patch.object(monitor, "_read_cpu_usage", side_effect=lambda: next(readings)),
        patch.object(monitor, "cpu_count", return_value=1.0),
        patch("livekit.agents.utils.hw.cpu.psutil.cpu_count", return_value=8),
        patch("livekit.agents.utils.hw.cpu.time.sleep", return_value=None),
        patch("livekit.agents.utils.hw.cpu.time.monotonic", side_effect=lambda: next(stamps)),
    ):
        assert monitor.cpu_percent(interval=0.5) == 0.0


def test_reads_process_cgroup_stat_with_limited_ancestor(monkeypatch):
    monitor = CGroupV2CPUMonitor()
    monkeypatch.delenv("NUM_CPUS", raising=False)

    child_path = "/sys/fs/cgroup/parent/child"
    child_stat_path = f"{child_path}/cpu.stat"
    cpu_stat_reads = iter(["usage_usec 0\n", "usage_usec 550000\n"])
    files = {
        "/proc/self/cgroup": "0::/parent/child\n",
        f"{child_path}/cpu.max": "max 100000\n",
        "/sys/fs/cgroup/parent/cpu.max": "100000 100000\n",
        "/sys/fs/cgroup/cpu.max": "max 100000\n",
    }
    opened: list[str] = []

    def open_file(path: str, *_args, **_kwargs):
        opened.append(path)
        if path == child_stat_path:
            return io.StringIO(next(cpu_stat_reads))
        try:
            return io.StringIO(files[path])
        except KeyError:
            raise FileNotFoundError(path) from None

    stamps = iter([0.0, 0.5])
    with (
        patch("builtins.open", side_effect=open_file),
        patch(
            "livekit.agents.utils.hw.cpu.os.path.exists",
            side_effect=lambda path: path == child_stat_path,
        ),
        patch("livekit.agents.utils.hw.cpu.psutil.cpu_count", return_value=8),
        patch("livekit.agents.utils.hw.cpu.time.sleep", return_value=None),
        patch("livekit.agents.utils.hw.cpu.time.monotonic", side_effect=lambda: next(stamps)),
    ):
        assert monitor.cpu_percent(interval=0.5) == 1.0

    assert opened.count(child_stat_path) == 2
    assert f"{child_path}/cpu.max" in opened
    assert "/sys/fs/cgroup/parent/cpu.max" in opened


def test_cpu_stat_path_falls_back_for_unavailable_or_escaping_paths():
    monitor = CGroupV2CPUMonitor()

    with (
        patch("builtins.open", mock_open(read_data="0::/host-only-path\n")),
        patch("livekit.agents.utils.hw.cpu.os.path.exists", return_value=False),
    ):
        assert monitor._cpu_stat_path() == "/sys/fs/cgroup/cpu.stat"

    with (
        patch("builtins.open", mock_open(read_data="0::/../../host-only-path\n")),
        patch("livekit.agents.utils.hw.cpu.os.path.exists") as exists,
    ):
        assert monitor._cpu_stat_path() == "/sys/fs/cgroup/cpu.stat"
        exists.assert_not_called()
