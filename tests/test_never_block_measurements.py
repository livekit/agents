"""Measurement contracts, including deliberately stalled and incomplete workloads."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import sys
import time
from pathlib import Path

import pytest
from scripts.bench import reports, sessions
from scripts.bench.metrics import (
    CallbackSlices,
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
        original_run = asyncio.Handle._run
        try:
            await monitor.ready.wait()
            # A deliberately bad integration must remain visible despite skipped deadlines.
            with CallbackSlices() as callbacks:
                asyncio.get_running_loop().call_soon(time.sleep, 0.025)
                await asyncio.sleep(0.03)
            assert distribution(monitor.samples).max_ms >= 20
            assert distribution(callbacks.wall).max_ms >= 20
            assert asyncio.Handle._run is original_run
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


@pytest.mark.no_concurrent
def test_recording_timing_observations_does_not_trigger_collection() -> None:
    capture = reports.ReportCapture()
    monitor = LoopLag()
    collections: list[int] = []

    def observe(phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            collections.append(info["generation"])

    gc.collect()
    gc.callbacks.append(observe)
    try:
        for index in range(5000):
            capture.sdk_slices.append(reports.SdkSlice(index, 1, index + 1, 1, "append_batch"))
            monitor.observations.append((index, index + 1))
    finally:
        gc.callbacks.remove(observe)

    assert not collections, "recording timings must not induce pauses in the measured workload"
    assert len(capture.sdk_slices) == len(monitor.observations) == 5000
    assert max(sample.cpu_ns for sample in capture.sdk_slices) == 5000
    assert max(lag for _, lag in monitor.observations) == 5000


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

    def provenance_command(command: list[str], *, text: bool = False) -> str | bytes:
        return "test-revision\n" if text else b""

    monkeypatch.setattr(sessions, "run", failed)
    monkeypatch.setattr(sessions.subprocess, "check_output", provenance_command)
    args = argparse.Namespace(output=output, sessions=[1], cadence_ms=[10])
    with pytest.raises(RuntimeError, match="incomplete playback"):
        asyncio.run(sessions.main(args))
    assert output.read_text() == "previous valid baseline"


@pytest.mark.asyncio
async def test_report_qualification_refuses_an_unqualified_idle_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failed_idle(*args: object) -> dict[str, object]:
        return {
            "workload_complete": True,
            "baseline_budget_pass": False,
            "measured_budget_pass": True,
        }

    monkeypatch.setattr(reports, "run", failed_idle)
    args = argparse.Namespace(
        output=tmp_path / "reports.json", sessions=[1], items=10_000, check=True
    )
    assert await reports.main(args) == 1


@pytest.mark.asyncio
async def test_invalid_report_limits_preserve_capture_without_starting_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(Path(reports.__file__).with_name("budgets.json").read_text())
    manifest["interfaces"]["session_report"]["limits"]["loop_lag_max_ms_exclusive"] = 50
    (tmp_path / "budgets.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(reports, "__file__", str(tmp_path / "reports.py"))
    output = tmp_path / "capture.json"
    output.write_text("previous valid capture")
    started = False

    async def work(*args: object) -> dict[str, object]:
        nonlocal started
        started = True
        return {
            "workload_complete": True,
            "baseline_budget_pass": True,
            "measured_budget_pass": True,
        }

    monkeypatch.setattr(reports, "run", work)
    args = argparse.Namespace(output=output, sessions=[1], items=10_000, check=True)
    with pytest.raises(ValueError, match="loop_lag_max_ms_exclusive"):
        await reports.main(args)
    assert not started
    assert output.read_text() == "previous valid capture"


def test_exceeded_frame_budgets_fail_even_when_average_work_is_fast() -> None:
    budget = reports.load_budget()
    healthy = distribution([100_000] * 1000)
    stalled = distribution([100_000] * 1000 + [10_000_000])
    assert cadence_budget_pass(healthy, 10, healthy)
    assert not cadence_budget_pass(stalled, 10, healthy)
    assert not cadence_budget_pass(distribution([2_000_000] * 1000), 10)
    assert not cadence_budget_pass(healthy, 10, distribution([2_000_000]))
    assert budget.scheduling_pass(healthy)
    assert not budget.scheduling_pass(stalled)
    assert not budget.scheduling_pass(distribution([2_000_000] * 1000))
    assert budget.cpu_pass(healthy)
    assert not budget.cpu_pass(stalled)


def test_report_qualification_uses_declared_limits_and_sample_count(tmp_path: Path) -> None:
    manifest = json.loads(Path(reports.__file__).with_name("budgets.json").read_text())
    report = manifest["interfaces"]["session_report"]
    report["workload"]["minimum_observations_per_distribution"] = 2000
    report["limits"]["loop_lag_p999_ms_exclusive"] = 0.25
    report["limits"]["callback_thread_cpu_max_ms"] = 0.4
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(manifest))
    budget = reports.load_budget(path)

    assert budget.scheduling_pass(distribution([100_000] * 2000))
    assert not budget.scheduling_pass(distribution([300_000] * 2000))
    assert not budget.scheduling_pass(distribution([100_000] * 1999))
    assert not budget.cpu_pass(distribution([500_000] * 2000))
    assert not budget.cpu_pass(distribution([100_000] * 1999))
