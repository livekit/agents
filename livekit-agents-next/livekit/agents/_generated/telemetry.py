# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs
#     rust/crates/lk-agents-abi/src/telemetry/work.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

LoopBlockSeverity: TypeAlias = Literal["warning", "error"]


@dataclass(frozen=True, kw_only=True, slots=True)
class BlockedReport:
    """A host monitor's rendered report. Its parent is an opaque SDK context used only by the host
    adapter; the policy's LoopBlock above contains no live objects.
    """

    duration: float
    started_at: float
    warn_threshold: float
    severity: LoopBlockSeverity
    gc_time: float
    cpu_time: float
    watchdog_gap: float
    process_descheduled: bool
    task_name: str | None
    stacks: list[str]
    importing: str | None = None
    parent_span_context: object | None = None
    location: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class LoopMonitorSpan:
    attributes: dict[str, object]
    error: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class LoopMonitorLog:
    message: str
    extra: dict[str, object]


@dataclass(frozen=True, kw_only=True, slots=True)
class LoopMonitorEmission:
    report: BlockedReport
    span: LoopMonitorSpan | None
    log: LoopMonitorLog | None
    reported: bool


TelemetryWorkOutcome: TypeAlias = Literal["incomplete_cleanup", "failed", "settled"]


@dataclass(frozen=True, kw_only=True, slots=True)
class TelemetryWorkObservation:
    """Runtime work identity and supplied monotonic time, without an SDK object or exception text."""

    key: str
    monotonic_ns: int
    outcome: TelemetryWorkOutcome
