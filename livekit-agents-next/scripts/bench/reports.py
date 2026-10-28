"""Measure session-report conversion with the shared never-block qualification harness."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from scripts.bench.metrics import CallbackSlices, LoopLag, cadence_budget_pass, distribution

from livekit.agents import _core, llm
from livekit.agents.job import SessionReport
from livekit.agents.telemetry import _destination, _export, _providers, traces


@dataclass(frozen=True)
class SdkSlice:
    started_ns: int
    wall_ns: int
    cpu_ns: int
    thread_id: int
    stage: str


@dataclass(frozen=True)
class GcPause:
    started_ns: int
    wall_ns: int
    cpu_ns: int
    thread_id: int
    generation: int
    collected: int


@dataclass
class ReportCapture:
    uploads: dict[str, str] = field(default_factory=dict)
    items: Counter[str] = field(default_factory=Counter)
    sdk_slices: list[SdkSlice] = field(default_factory=list)
    gc_pauses: list[GcPause] = field(default_factory=list)
    before_diagnostics: dict[str, int] = field(default_factory=dict)
    after_diagnostics: dict[str, int] = field(default_factory=dict)


class _CountingExporter(LogRecordExporter):
    def __init__(self, capture: ReportCapture) -> None:
        self.capture = capture

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        for record in batch:
            if record.log_record.body == "chat item":
                attributes = record.log_record.attributes or {}
                job_id = attributes.get("job_id")
                if isinstance(job_id, str):
                    self.capture.items[job_id] += 1
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class _UploadSink:
    def __init__(self, capture: ReportCapture, native: _core._core_SessionReportUpload) -> None:
        self.capture = capture
        self.native = native

    def content_policy(
        self, transcript: bool, audio: bool, content: bool, redaction: bool
    ) -> tuple[bool, bool]:
        return self.native.content_policy(transcript, audio, content, redaction)

    async def upload(
        self,
        base_url: str,
        room_id: str,
        job_id: str,
        simulated: bool,
        redaction_enabled: bool,
        start_time_millis: int,
        transcript: str | None,
        audio_path: str | None,
        disabled: bool,
    ) -> bool:
        if transcript is not None:
            self.capture.uploads[job_id] = transcript
        return True


class _ObservedWorker:
    def __init__(
        self,
        capture: ReportCapture,
        native: _core._core_TelemetryWorker,
        stages: dict[str, str],
    ) -> None:
        self.capture = capture
        self.native = native
        self.stages = stages

    def reserve(self, key: str, retained_bytes: int) -> bool:
        return self.native.reserve(key, retained_bytes)

    def release(self, key: str) -> None:
        self.native.release(key)

    def record_drop(self, kind: str, count: int) -> None:
        self.native.record_drop(kind, count)

    async def wait_async(self, key: str, timeout: float) -> bool:
        return await self.native.wait_async(key, timeout)

    def submit(self, key: str, callback: Callable[[], object], retained_bytes: int) -> bool:
        stage = self.stages.get(key, key)

        def measured() -> object:
            started = time.monotonic_ns()
            cpu = time.thread_time_ns()
            try:
                return callback()
            finally:
                self.capture.sdk_slices.append(
                    SdkSlice(
                        started,
                        time.monotonic_ns() - started,
                        time.thread_time_ns() - cpu,
                        threading.get_ident(),
                        stage,
                    )
                )

        return self.native.submit(key, measured, retained_bytes)


@contextlib.contextmanager
def capture_reports() -> Iterator[ReportCapture]:
    """Use the real native preparation and SDK; replace only the final network destination."""
    capture = ReportCapture()
    native_upload = _core._core_SessionReportUpload("key", "secret")
    native_worker = _core._core_TelemetryWorker()
    capture.before_diagnostics = dict(native_worker.diagnostics)
    native_call = _export.call_bounded
    stages: dict[str, str] = {}
    gc_start: tuple[int, int] | None = None

    def observe_gc(phase: str, info: dict[str, int]) -> None:
        nonlocal gc_start
        if phase == "start":
            gc_start = (time.monotonic_ns(), time.thread_time_ns())
        elif gc_start is not None:
            started, cpu = gc_start
            capture.gc_pauses.append(
                GcPause(
                    started,
                    time.monotonic_ns() - started,
                    time.thread_time_ns() - cpu,
                    threading.get_ident(),
                    info["generation"],
                    info["collected"],
                )
            )
            gc_start = None

    async def observed_call(
        key: str, callback: Callable[[], object], retained_bytes: int, timeout: float
    ) -> object:
        function = callback.func if isinstance(callback, partial) else callback
        stages[key] = function.__name__
        try:
            return await native_call(key, callback, retained_bytes, timeout)
        finally:
            stages.pop(key, None)

    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(SimpleLogRecordProcessor(_CountingExporter(capture)))
    cloud = _providers._CloudTelemetry()
    cloud._logger_provider = provider
    with contextlib.ExitStack() as stack:
        stack.callback(provider.shutdown)
        stack.enter_context(
            patch.dict(os.environ, LIVEKIT_API_KEY="key", LIVEKIT_API_SECRET="secret")
        )
        upload_sink = _UploadSink(capture, native_upload)
        stack.enter_context(
            patch.object(
                _core, "_core_SessionReportUpload", lambda api_key, api_secret: upload_sink
            )
        )
        observed_worker = _ObservedWorker(capture, native_worker, stages)
        stack.enter_context(patch.object(_core, "_core_TelemetryWorker", lambda: observed_worker))
        stack.enter_context(patch.object(_export, "worker", lambda: observed_worker))
        stack.enter_context(patch.object(_export, "call_bounded", observed_call))
        stack.enter_context(patch.object(traces, "_cloud", cloud))
        stack.enter_context(patch.object(_destination, "upload_gate", _core.UploadGate()))
        gc.callbacks.append(observe_gc)
        stack.callback(gc.callbacks.remove, observe_gc)
        try:
            yield capture
        finally:
            capture.after_diagnostics = dict(native_worker.diagnostics)


def make_report(items: list[llm.ChatItem], job_id: str = "report-job") -> SessionReport:
    return SessionReport(
        job_id=job_id,
        room_id="report-room",
        room="report-room-name",
        options={
            "recording_options": {
                "audio": False,
                "traces": False,
                "logs": False,
                "transcript": True,
                "redaction": False,
            }
        },
        events=[],
        chat_history=llm.ChatContext(items=items),
        _capture_content=True,
    )


async def prepare(report: SessionReport) -> None:
    await traces._upload_session_report(
        agent_name="report-agent",
        observability_url="https://unused.invalid",
        report=report,
        tagger=_core.Tagger(),
        http_session=SimpleNamespace(),
    )


async def run(count: int, item_count: int) -> dict[str, object]:
    items = [llm.ChatMessage(role="user", content=["hello"], id=str(i)) for i in range(item_count)]
    reports = [make_report(items, f"report-{i}") for i in range(count)]
    monitor = LoopLag()
    pulse = asyncio.create_task(monitor.run())
    try:
        with capture_reports() as capture, CallbackSlices() as callbacks:
            await monitor.ready.wait()
            idle_started = time.monotonic_ns()
            idle_cpu_started = time.thread_time_ns()
            while len(monitor.samples) < 1_000:
                await asyncio.sleep(0.01)
            baseline = distribution(monitor.samples)
            baseline_wall = time.monotonic_ns() - idle_started
            baseline_cpu = time.thread_time_ns() - idle_cpu_started
            monitor.samples.clear()
            monitor.observations.clear()
            callbacks.values = callbacks.values[:0]
            started = time.monotonic()
            workload_started = time.monotonic_ns()
            results = await asyncio.gather(
                *(prepare(report) for report in reports), return_exceptions=True
            )
            workload_finished = time.monotonic_ns()
            elapsed = time.monotonic() - started
            while len(monitor.samples) < 1_000:
                await asyncio.sleep(0.01)
        lag = distribution(monitor.samples)
        cpu = distribution([item.cpu_ns for item in capture.sdk_slices])
        wall = distribution([item.wall_ns for item in capture.sdk_slices])
        loop_cpu = distribution(callbacks.cpu)
        completed = len(capture.uploads)
        refused = count - completed
        refusal_observations = (
            capture.after_diagnostics["dropped"] - capture.before_diagnostics["dropped"]
        )
        complete = (
            completed > 0
            and refusal_observations >= refused
            and all(
                len(json.loads(transcript)["items"]) == item_count
                and capture.items[job_id] == item_count
                for job_id, transcript in capture.uploads.items()
            )
        )
        observed, worst_lag = max(monitor.observations, key=lambda pair: pair[1])
        overlapping = [
            asdict(item)
            for item in capture.sdk_slices
            if item.started_ns <= observed
            and item.started_ns + item.wall_ns >= observed - worst_lag
        ]
        cpu_pass = all(
            samples.count >= 1_000 and samples.p999_ms <= 1 and samples.max_ms <= 2
            for samples in (cpu, loop_cpu)
        )
        largest_cpu = sorted(capture.sdk_slices, key=lambda sample: sample.cpu_ns, reverse=True)[
            :10
        ]
        return {
            "sessions": count,
            "history_items": item_count,
            "seconds": elapsed,
            "completed": completed,
            "refused": refused,
            "refusal_observations": refusal_observations,
            "refused_or_failed": [
                str(result) for result in results if isinstance(result, BaseException)
            ],
            "baseline_lag": asdict(baseline),
            "baseline_wall_ms": baseline_wall / 1e6,
            "baseline_interval_ns": [idle_started, idle_started + baseline_wall],
            "workload_interval_ns": [workload_started, workload_finished],
            "baseline_thread_cpu_ms": baseline_cpu / 1e6,
            "cpu_count": os.cpu_count(),
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "loop_lag": asdict(lag),
            "sdk_callback_cpu": asdict(cpu),
            "sdk_callback_wall": asdict(wall),
            "loop_callback_cpu": asdict(loop_cpu),
            "loop_callback_wall": asdict(distribution(callbacks.wall)),
            "largest_cpu_loop_slices": callbacks.largest("cpu"),
            "largest_wall_loop_slices": callbacks.largest("wall"),
            "largest_lag_overlapping_sdk_slices": overlapping,
            "largest_lag": {"observed_ns": observed, "lag_ns": worst_lag},
            "largest_cpu_sdk_slices": [asdict(sample) for sample in largest_cpu],
            "largest_cpu_overlapping_gc": [
                asdict(pause)
                for pause in capture.gc_pauses
                if any(
                    pause.thread_id == sample.thread_id
                    and pause.started_ns < sample.started_ns + sample.wall_ns
                    and pause.started_ns + pause.wall_ns > sample.started_ns
                    for sample in largest_cpu
                )
            ],
            "gc_pauses": [asdict(pause) for pause in capture.gc_pauses],
            "stages": {
                stage: asdict(
                    distribution([s.cpu_ns for s in capture.sdk_slices if s.stage == stage])
                )
                for stage in {sample.stage for sample in capture.sdk_slices}
            },
            "workload_complete": complete,
            "baseline_budget_pass": cadence_budget_pass(baseline, 10),
            "measured_budget_pass": complete and cadence_budget_pass(lag, 10) and cpu_pass,
        }
    finally:
        pulse.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pulse


async def main(args: argparse.Namespace) -> int:
    cases = [await run(count, args.items) for count in args.sessions]
    result = {"cases": cases}
    if not all(case["workload_complete"] for case in cases):
        raise RuntimeError("report workload produced no complete report")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return int(
        args.check
        and not all(case["baseline_budget_pass"] and case["measured_budget_pass"] for case in cases)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions", nargs="+", type=int, choices=(1, 8, 32, 64), default=[1, 8, 32, 64]
    )
    parser.add_argument("--items", type=int, choices=(10, 1_000, 10_000), default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
