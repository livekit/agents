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
from array import array
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args
from unittest.mock import patch

from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from pydantic import AliasPath, BaseModel, ConfigDict, Field

from livekit.agents import _core, llm
from livekit.agents.job import SessionReport
from livekit.agents.telemetry import _destination, _providers, traces
from scripts.bench.metrics import (
    CallbackSlices,
    Distribution,
    LoopLag,
    distribution,
)

BUDGETS_PATH = Path(__file__).resolve().parents[2] / "rust" / "bench-budgets.json"

_BUDGET_CONFIG = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)
HistoryItems = Literal[10, 1_000, 10_000]


class ReportWorkload(BaseModel):
    model_config = _BUDGET_CONFIG

    sessions: list[Literal[1, 8, 32, 64]] = Field(min_length=1)
    history_items: HistoryItems
    minimum_observations_per_distribution: int = Field(ge=1_000)
    heartbeat_ms: float = Field(gt=0, le=1)
    control_cadence_ms: Literal[10]


class ReportLimits(BaseModel):
    model_config = _BUDGET_CONFIG

    loop_lag_p999_ms_exclusive: float = Field(gt=0, le=2)
    loop_lag_max_ms_exclusive: float = Field(gt=0, le=5)
    callback_thread_cpu_p999_ms: float = Field(gt=0, le=1)
    callback_thread_cpu_max_ms: float = Field(gt=0, le=2)


class ReportBudget(BaseModel):
    model_config = _BUDGET_CONFIG

    workload: ReportWorkload = Field(
        validation_alias=AliasPath("interfaces", "session_report", "workload")
    )
    limits: ReportLimits = Field(
        validation_alias=AliasPath("interfaces", "session_report", "limits")
    )

    def scheduling_pass(self, samples: Distribution) -> bool:
        return (
            samples.count >= self.workload.minimum_observations_per_distribution
            and samples.p999_ms < self.limits.loop_lag_p999_ms_exclusive
            and samples.max_ms < self.limits.loop_lag_max_ms_exclusive
        )

    def cpu_pass(self, samples: Distribution) -> bool:
        return (
            samples.count >= self.workload.minimum_observations_per_distribution
            and samples.p999_ms <= self.limits.callback_thread_cpu_p999_ms
            and samples.max_ms <= self.limits.callback_thread_cpu_max_ms
        )


def load_budget(path: Path | None = None) -> ReportBudget:
    return ReportBudget.model_validate_json((path or BUDGETS_PATH).read_bytes())


@dataclass(frozen=True)
class SdkSlice:
    started_ns: int
    wall_ns: int
    cpu_ns: int
    thread_id: int
    stage: str


class SdkSamples:
    """Retain numeric measurements without adding one GC-tracked object per callback."""

    def __init__(self) -> None:
        self._values = array("Q")
        self._stages: list[str] = []
        self._stage_ids: dict[str, int] = {}

    def append(
        self, started_ns: int, wall_ns: int, cpu_ns: int, thread_id: int, stage: str
    ) -> None:
        stage_id = self._stage_ids.get(stage)
        if stage_id is None:
            stage_id = len(self._stages)
            self._stage_ids[stage] = stage_id
            self._stages.append(stage)
        self._values.append(started_ns)
        self._values.append(wall_ns)
        self._values.append(cpu_ns)
        self._values.append(thread_id)
        self._values.append(stage_id)

    def __len__(self) -> int:
        return len(self._values) // 5

    def __iter__(self) -> Iterator[SdkSlice]:
        for at in range(0, len(self._values), 5):
            yield SdkSlice(
                self._values[at],
                self._values[at + 1],
                self._values[at + 2],
                self._values[at + 3],
                self._stages[self._values[at + 4]],
            )


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
    sdk_slices: SdkSamples = field(default_factory=SdkSamples)
    gc_pauses: list[GcPause] = field(default_factory=list)


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
        *,
        timeout: float | None = None,
    ) -> bool:
        if transcript is not None:
            self.capture.uploads[job_id] = transcript
        return True


@contextlib.contextmanager
def capture_reports() -> Iterator[ReportCapture]:
    """Use the real native preparation and SDK; replace only the final network destination."""
    capture = ReportCapture()
    native_upload = _core._core_SessionReportUpload("key", "secret")
    to_thread = asyncio.to_thread
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

    async def observed_to_thread(callback: Callable[..., object], /, *args: object) -> object:
        def measured() -> object:
            started = time.monotonic_ns()
            cpu = time.thread_time_ns()
            try:
                return callback(*args)
            finally:
                capture.sdk_slices.append(
                    started,
                    time.monotonic_ns() - started,
                    time.thread_time_ns() - cpu,
                    threading.get_ident(),
                    getattr(callback, "__name__", "callback"),
                )

        return await to_thread(measured)

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
        stack.enter_context(patch.object(asyncio, "to_thread", observed_to_thread))
        stack.enter_context(patch.object(traces, "_cloud", cloud))
        stack.enter_context(patch.object(_destination, "upload_gate", _core.UploadGate()))
        gc.callbacks.append(observe_gc)
        stack.callback(gc.callbacks.remove, observe_gc)
        yield capture


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


async def run(count: int, item_count: int, budget: ReportBudget | None = None) -> dict[str, object]:
    budget = budget or load_budget()
    items = [llm.ChatMessage(role="user", content=["hello"], id=str(i)) for i in range(item_count)]
    reports = [make_report(items, f"report-{i}") for i in range(count)]
    monitor = LoopLag(period_ms=budget.workload.heartbeat_ms)
    pulse = asyncio.create_task(monitor.run())
    try:
        with capture_reports() as capture, CallbackSlices() as callbacks:
            await monitor.ready.wait()
            idle_started = time.monotonic_ns()
            idle_cpu_started = time.thread_time_ns()
            while len(monitor.samples) < budget.workload.minimum_observations_per_distribution:
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
            while len(monitor.samples) < budget.workload.minimum_observations_per_distribution:
                await asyncio.sleep(0.01)
        lag = distribution(monitor.samples)
        cpu = distribution([item.cpu_ns for item in capture.sdk_slices])
        wall = distribution([item.wall_ns for item in capture.sdk_slices])
        loop_cpu = distribution(callbacks.cpu)
        completed = len(capture.uploads)
        refused = count - completed
        complete = completed > 0 and all(
            len(json.loads(transcript)["items"]) == item_count
            and capture.items[job_id] == item_count
            for job_id, transcript in capture.uploads.items()
        )
        observed, worst_lag = max(monitor.observations, key=lambda pair: pair[1])
        overlapping = [
            asdict(item)
            for item in capture.sdk_slices
            if item.started_ns <= observed
            and item.started_ns + item.wall_ns >= observed - worst_lag
        ]
        cpu_pass = all(budget.cpu_pass(samples) for samples in (cpu, loop_cpu))
        largest_cpu = sorted(capture.sdk_slices, key=lambda sample: sample.cpu_ns, reverse=True)[
            :10
        ]
        return {
            "sessions": count,
            "history_items": item_count,
            "control_cadence_ms": budget.workload.control_cadence_ms,
            "seconds": elapsed,
            "completed": completed,
            "refused": refused,
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
            "baseline_budget_pass": budget.scheduling_pass(baseline),
            "measured_budget_pass": complete and budget.scheduling_pass(lag) and cpu_pass,
        }
    finally:
        pulse.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pulse


async def main(args: argparse.Namespace | None = None, *, budget_path: Path | None = None) -> int:
    budget = load_budget(budget_path)
    args = args if args is not None else parse_args(budget)
    cases = [await run(count, args.items, budget) for count in args.sessions]
    result = {"cases": cases}
    if not all(case["workload_complete"] for case in cases):
        raise RuntimeError("report workload produced no complete report")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return int(
        args.check
        and not all(case["baseline_budget_pass"] and case["measured_budget_pass"] for case in cases)
    )


def parse_args(budget: ReportBudget) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        choices=budget.workload.sessions,
        default=budget.workload.sessions,
    )
    parser.add_argument(
        "--items", type=int, choices=get_args(HistoryItems), default=budget.workload.history_items
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
