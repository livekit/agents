"""Measure session-report conversion with the shared never-block qualification harness."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from scripts.bench.metrics import LoopLag, cadence_budget_pass, distribution

from livekit.agents import _core, llm
from livekit.agents.job import SessionReport
from livekit.agents.telemetry import _destination, _export, _providers, traces


@dataclass(frozen=True)
class SdkSlice:
    started_ns: int
    wall_ns: int
    cpu_ns: int


@dataclass
class ReportCapture:
    uploads: dict[str, str] = field(default_factory=dict)
    items: Counter[str] = field(default_factory=Counter)
    sdk_slices: list[SdkSlice] = field(default_factory=list)
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


@contextlib.contextmanager
def capture_reports() -> Iterator[ReportCapture]:
    """Use the real native preparation and SDK; replace only the final network destination."""
    capture = ReportCapture()
    native_upload = _core._core_SessionReportUpload("key", "secret")
    native_worker = _core._core_TelemetryWorker()
    capture.before_diagnostics = dict(native_worker.diagnostics)

    class UploadSink:
        def __init__(self, api_key: str, api_secret: str) -> None:
            pass

        def content_policy(
            self, transcript: bool, audio: bool, content: bool, redaction: bool
        ) -> tuple[bool, bool]:
            return native_upload.content_policy(transcript, audio, content, redaction)

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
                capture.uploads[job_id] = transcript
            return True

    class ObservedWorker:
        def reserve(self, key: str, retained_bytes: int) -> bool:
            return native_worker.reserve(key, retained_bytes)

        def release(self, key: str) -> None:
            native_worker.release(key)

        def record_drop(self, kind: str, count: int) -> None:
            native_worker.record_drop(kind, count)

        async def wait_async(self, key: str, timeout: float) -> bool:
            return await native_worker.wait_async(key, timeout)

        def submit(self, key: str, callback: Callable[[], object], retained_bytes: int) -> bool:
            def measured() -> object:
                started = time.monotonic_ns()
                cpu = time.thread_time_ns()
                try:
                    return callback()
                finally:
                    capture.sdk_slices.append(
                        SdkSlice(
                            started, time.monotonic_ns() - started, time.thread_time_ns() - cpu
                        )
                    )

            return native_worker.submit(key, measured, retained_bytes)

    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(SimpleLogRecordProcessor(_CountingExporter(capture)))
    cloud = _providers._CloudTelemetry()
    cloud._logger_provider = provider
    with contextlib.ExitStack() as stack:
        stack.callback(provider.shutdown)
        stack.enter_context(
            patch.dict(os.environ, LIVEKIT_API_KEY="key", LIVEKIT_API_SECRET="secret")
        )
        stack.enter_context(patch.object(traces, "capture_content_enabled", lambda: True))
        stack.enter_context(patch.object(_core, "_core_SessionReportUpload", UploadSink))
        observed_worker = ObservedWorker()
        stack.enter_context(patch.object(_core, "_core_TelemetryWorker", ObservedWorker))
        stack.enter_context(patch.object(_export, "worker", lambda: observed_worker))
        stack.enter_context(patch.object(traces, "_cloud", cloud))
        stack.enter_context(patch.object(_destination, "upload_gate", _core.UploadGate()))
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
        with capture_reports() as capture:
            await monitor.ready.wait()
            await asyncio.sleep(0.5)
            baseline = distribution(monitor.samples)
            monitor.samples.clear()
            monitor.observations.clear()
            started = time.monotonic()
            results = await asyncio.gather(
                *(prepare(report) for report in reports), return_exceptions=True
            )
            elapsed = time.monotonic() - started
            await asyncio.sleep(0)
        lag = distribution(monitor.samples)
        cpu = distribution([item.cpu_ns for item in capture.sdk_slices])
        wall = distribution([item.wall_ns for item in capture.sdk_slices])
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
        cpu_pass = cpu.p999_ms <= 1 and cpu.max_ms <= 2
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
            "loop_lag": asdict(lag),
            "sdk_callback_cpu": asdict(cpu),
            "sdk_callback_wall": asdict(wall),
            "largest_lag_overlapping_sdk_slices": overlapping,
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
    return int(args.check and not all(case["measured_budget_pass"] for case in cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions", nargs="+", type=int, choices=(1, 8, 32, 64), default=[1, 8, 32, 64]
    )
    parser.add_argument("--items", type=int, choices=(10, 1_000, 10_000), default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
