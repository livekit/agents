"""Measurement contracts, including deliberately stalled and incomplete workloads."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from pathlib import Path

import pytest
from scripts.bench import sessions
from scripts.bench.metrics import (
    LoopLag,
    NativeCalls,
    cadence_budget_pass,
    distribution,
    sampled_stacks,
)

pytestmark = pytest.mark.unit


def test_loop_monitor_retains_a_blocking_callback_stall() -> None:
    async def workload() -> None:
        monitor = LoopLag()
        heartbeat = asyncio.create_task(monitor.run())
        try:
            await monitor.ready.wait()
            # A deliberately bad integration must remain visible despite skipped deadlines.
            time.sleep(0.025)
            await asyncio.sleep(0.01)
            assert distribution(monitor.samples).max_ms >= 20
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    asyncio.run(workload())


def test_profiler_counts_visible_payload_materialization_and_restores_itself() -> None:
    with pytest.raises(RuntimeError, match="workload failed"):
        with NativeCalls() as profiler:
            assert memoryview(b"pcm payload").tobytes() == b"pcm payload"
            raise RuntimeError("workload failed")
    assert profiler.copy_calls == 1
    assert profiler.copy_bytes == 11
    assert sys.getprofile() is None


def test_native_sampling_preserves_worker_identity_and_inclusive_counts() -> None:
    observations = sampled_stacks(
        "    50 Thread_1: tokio-rt-worker\n"
        "    + 30 lk_agents_voice::runtime::Runtime::handle (in core)\n"
        "    +   20 pthread_mutex_lock (in libsystem)\n"
        "    50 Thread_2: tokio-rt-worker\n"
        "    + 50 tokio::runtime::park (in core)\n"
    )
    assert any(s.samples == 30 and s.kind == "core_stack" for s in observations)
    assert any(s.samples == 20 and "Thread_1" in s.thread for s in observations)
    assert any(s.samples == 50 and "Thread_2" in s.thread for s in observations)


@pytest.mark.parametrize("value", ["0", "-1", "257"])
def test_empty_or_excessive_session_counts_fail_before_execution(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        sessions.positive(value)


@pytest.mark.parametrize("value", ["0", "nan", "inf", "121"])
def test_invalid_workload_duration_fails_before_execution(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        sessions.duration(value)


def test_failed_workload_never_replaces_existing_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "baseline.json"
    output.write_text("previous valid baseline")

    async def failed(*args: object) -> dict[str, object]:
        raise RuntimeError("incomplete playback")

    monkeypatch.setattr(sessions, "run", failed)
    args = argparse.Namespace(output=output, sessions=[1], cadence_ms=[10])
    with pytest.raises(RuntimeError, match="incomplete playback"):
        asyncio.run(sessions.main(args))
    assert output.read_text() == "previous valid baseline"


def test_exceeded_frame_budgets_fail_even_when_average_work_is_fast() -> None:
    healthy = distribution([100_000] * 1000)
    stalled = distribution([100_000] * 1000 + [10_000_000])
    assert cadence_budget_pass(healthy, 10, healthy)
    assert not cadence_budget_pass(stalled, 10, healthy)
    assert not cadence_budget_pass(healthy, 10, distribution([2_000_000]))
