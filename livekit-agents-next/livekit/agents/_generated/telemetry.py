# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/metrics/otel.rs
#     rust/crates/lk-agents-abi/src/telemetry/policy.rs
#     rust/crates/lk-agents-abi/src/telemetry/report.rs
#     rust/crates/lk-agents-abi/src/telemetry/report/projection.rs
#     rust/crates/lk-agents-abi/src/telemetry/report/records.rs
#     rust/crates/lk-agents-abi/src/telemetry/loop_monitor.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .._core import ChatItem
    from .usage import ModelUsage

OtelValue: TypeAlias = int | float
"""Token counts stay integers: the exported data point type follows the first value seen."""


TelemetryValueKind: TypeAlias = Literal["scalar", "integer", "string", "bytes", "mapping", "sequence", "unsupported"]


TelemetryRecordKind: TypeAlias = Literal["span", "log", "event", "link", "scope", "value"]


TelemetryExportAction: TypeAlias = Literal["export", "hold", "drop", "capacity", "disabled"]


@dataclass(frozen=True, kw_only=True, slots=True)
class RecordingOptions:
    """`RecordingOptions`, the four keys `recording_enabled` reads plus `redaction`.

    `redaction = False` never disables project-level redaction; it only declines to ask for it.
    """

    audio: bool = False
    traces: bool = False
    logs: bool = False
    transcript: bool = False
    redaction: bool = False
    """Not part of `_RECORDING_OPTION_KEYS` -- `recording_enabled` ignores it."""


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluationRecord:
    """One entry of `Tagger.evaluations` (`observability.py:120`)."""

    name: str
    tag: str
    verdict: str
    reasoning: str
    instructions: str


Outcome: TypeAlias = Literal["success", "fail"]
"""The session outcome. `"fail"` bumps the outcome record's severity."""


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionReportInput:
    """`SessionReport._core_view()`: what the report's upload records are built from."""

    job_id: str
    room_id: str
    room: str
    agent_name: str
    sdk_version: str
    started_at: float | None
    timestamp: float
    options: object
    recording_options: RecordingOptions
    model_usage: list[ModelUsage]
    chat_history: list[ChatItem]
    omitted_events: int = 0


@dataclass(frozen=True, kw_only=True, slots=True)
class SpanUpdateWire:
    """`{"type": "span_update", "span": .., "attrs": {..}}`."""

    span: int
    """The span, as a `HostRef` id the host resolves."""
    attrs: SpanUpdateAttrsWire
    type: Literal["span_update"] = "span_update"


@dataclass(frozen=True, kw_only=True, slots=True)
class SpanUpdateAttrsWire:
    """The nested `attrs` dict: the five values `_apply_span_update` writes onto the span. A wire twin
    of `lk_agents_inference::interruption::SpanAttributes`, whose `&'static str` field the
    generator cannot map.
    """

    is_interruption: str
    probability: float
    total_duration: float
    prediction_duration: float
    detection_delay: float


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionReportSnapshot:
    """The serialized SDK snapshot. Native projection keeps its schema stable when content is denied."""

    job_id: str
    room_id: str
    room: str
    events: list[object]
    omitted_events: int = 0
    audio_recording_path: str | None
    audio_recording_started_at: float | None
    options: object | None
    chat_history: object | None
    timestamp: float
    usage: list[ModelUsage] | None
    sdk_version: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ProjectedReportEvent:
    """Captured once at the destination boundary; emission does not decode or project it again."""

    value: object
    content: bool
    retained_bytes: int


@dataclass(frozen=True, kw_only=True, slots=True)
class LogRecordWire:
    """A [`LogRecord`] as the dict `_upload_session_report` hands `logger.emit`: one integer of epoch
    nanos, where the log handler's records carry an `HrTime` pair.
    """

    logger: str
    body: str
    timestamp_nanos: int
    severity_number: int
    severity_text: str
    attributes: dict[str, object]


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
