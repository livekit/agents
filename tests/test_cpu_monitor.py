import logging
import time

import psutil
import pytest

from livekit.agents import utils
from livekit.agents.utils.hw.cpu import CGroupV2CPUMonitor

pytestmark = pytest.mark.unit

HOST_CPUS = 8
INTERVAL = 0.5
# one idle 0.5 s sample on the reporter's host, about 0.5 % of 8 CPUs
IDLE_USAGE_USEC = 21_600

# raw (start, end) usage_usec pairs logged on the affected Xen HVM guest
TORN_READS = [
    (5260817467, 2548880500),
    (5268770613, 5860175581),
    (5889766280, 5270782758),
    (1210759425, 5271033390),
    (5271381786, 5798443673),
    (5288672329, 1215868554),
]


@pytest.fixture
def monitor(monkeypatch: pytest.MonkeyPatch) -> CGroupV2CPUMonitor:
    monkeypatch.delenv("NUM_CPUS", raising=False)
    monkeypatch.setattr(psutil, "cpu_count", lambda: HOST_CPUS)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monitor = CGroupV2CPUMonitor()
    monkeypatch.setattr(monitor, "_read_cpu_max", lambda: ("max", 100000))
    return monitor


def sample(
    monitor: CGroupV2CPUMonitor,
    monkeypatch: pytest.MonkeyPatch,
    usage_start: int,
    usage_end: int,
    *,
    elapsed: float = INTERVAL,
) -> float:
    reads = iter([usage_start, usage_end])
    clock = iter([0.0, elapsed])
    monkeypatch.setattr(monitor, "_read_cpu_usage", lambda: next(reads))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    return monitor.cpu_percent(INTERVAL)


@pytest.mark.parametrize("elapsed", [0.5, 0.6])
def test_normal_delta_uses_measured_elapsed(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch, elapsed: float
) -> None:
    # one cpu-second of usage over the measured interval on 8 CPUs
    pct = sample(monitor, monkeypatch, 5_000_000_000, 5_001_000_000, elapsed=elapsed)
    assert pct == pytest.approx(1.0 / (elapsed * HOST_CPUS))


def test_negative_delta_holds_previous_sample(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    good = sample(monitor, monkeypatch, 5_000_000_000, 5_001_000_000)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        pct = sample(monitor, monkeypatch, *TORN_READS[0])
    assert pct == good
    assert "impossible" in caplog.text


def test_over_ceiling_delta_holds_previous_sample(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    good = sample(monitor, monkeypatch, 5_000_000_000, 5_001_000_000)
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        pct = sample(monitor, monkeypatch, *TORN_READS[1])
    assert pct == good
    assert pct != 1.0
    assert "impossible" in caplog.text


def test_first_discarded_sample_reads_idle(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sample(monitor, monkeypatch, *TORN_READS[0]) == 0.0


def test_reporter_pattern_stays_below_threshold(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    idle = 5_270_000_000
    reads: list[tuple[int, int]] = []
    for torn in [None, None, *TORN_READS[:3], None, *TORN_READS[3:], None, None]:
        reads.append(torn or (idle, idle + IDLE_USAGE_USEC))
        idle += IDLE_USAGE_USEC

    avg = utils.MovingAverage(5)
    idle_pct = IDLE_USAGE_USEC / 1_000_000 / (INTERVAL * HOST_CPUS)
    for usage_start, usage_end in reads:
        avg.add_sample(sample(monitor, monkeypatch, usage_start, usage_end))
        assert avg.get_avg() == pytest.approx(idle_pct)


def test_delta_at_ceiling_is_full_load(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = int(INTERVAL * HOST_CPUS * 1_000_000)
    assert sample(monitor, monkeypatch, 5_000_000_000, 5_000_000_000 + full) == 1.0


def test_burst_above_quota_clamps_instead_of_discarding(
    monitor: CGroupV2CPUMonitor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # a 2-CPU cpu.max quota on an 8-CPU host, bursting to 4 CPUs for the interval
    monkeypatch.setattr(monitor, "_read_cpu_max", lambda: ("200000", 100000))
    assert monitor.cpu_count() == 2.0
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        pct = sample(monitor, monkeypatch, 5_000_000_000, 5_002_000_000)
    assert pct == 1.0
    assert "impossible" not in caplog.text
