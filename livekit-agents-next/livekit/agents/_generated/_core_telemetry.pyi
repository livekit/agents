# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Sequence

from .._core import ChatItem
from .telemetry import (
    LogRecordWire,
    ProjectedReportEvent,
    RecordingOptions,
    SessionReportSnapshot,
    TelemetryExportAction,
    TelemetryRecordKind,
    TelemetryValueKind,
)

class _core_TelemetryPolicy:
    """The host supplies SDK fields and lengths; shared policy decides what may be retained/exported."""
    def __init__(self, allow_pii: bool, capture_content: bool, project_redaction: bool) -> None: ...
    @property
    def redacted(self) -> bool: ...
    @property
    def max_depth(self) -> int: ...
    @property
    def max_collection(self) -> int: ...
    @property
    def max_events(self) -> int: ...
    @property
    def max_record_bytes(self) -> int: ...
    @property
    def max_pending(self) -> int: ...
    @property
    def max_records(self) -> int: ...
    @property
    def batch_items(self) -> int: ...
    @property
    def schedule_delay_millis(self) -> int: ...
    @property
    def export_timeout_millis(self) -> int: ...
    @property
    def max_providers(self) -> int: ...
    @property
    def shutdown_timeout_seconds(self) -> float: ...
    @property
    def report_retained_bytes(self) -> int: ...
    @property
    def upload_timeout_seconds(self) -> float: ...
    def export_action(
        self,
        registered: bool,
        signal_enabled: bool,
        job_open: bool,
        held_count: int,
        recording_disabled: bool,
    ) -> TelemetryExportAction:
        """The destination may export, hold, drop, refuse capacity or stop after recording is disabled."""
    def is_pii_attribute(self, key: str) -> bool: ...
    def attribute_action(
        self,
        parent: str | None,
        key: str,
        destination: bool,
    ) -> tuple[bool, str | None]:
        """Whether to keep the field and, if present, the replacement for its value."""
    def keep_event(self, name: str) -> bool: ...
    def log_body(self, safe_body: bool, body: str | None) -> str | None:
        """A replacement body, or None to snapshot the original SDK value."""
    def status_description(self, is_error: bool, description: str | None) -> str | None: ...
    def value_bytes(
        self,
        kind: TelemetryValueKind,
        length: int,
        depth: int,
        children: Sequence[int],
    ) -> int:
        """Conservative retained bytes for an SDK scalar or collection. Child charges are already
        bounded verdicts; unsupported kinds, depth and collection overflow refuse the record.
        """
    def record_bytes(
        self,
        kind: TelemetryRecordKind,
        fields: Sequence[int],
        event_count: int,
        link_count: int,
    ) -> int:
        """SDK record overhead plus field charges. Counts are supplied before traversing events/links."""

class _core_MetricSeries:
    """The framework's metric series. Subject folds decide measurements; this refuses unbounded labels
    and series before the host SDK creates a new one.
    """
    def __init__(self) -> None: ...
    def admit(self, name: str, attributes: Sequence[tuple[str, str]]) -> bool: ...

class _core_SessionReportProjection:
    def __init__(
        self,
        capture_content: bool,
        options: RecordingOptions,
        project_redaction: bool,
    ) -> None: ...
    @property
    def content(self) -> bool: ...
    @property
    def transcript(self) -> bool: ...
    @property
    def audio(self) -> bool: ...
    def event_limit(self, items: int) -> int:
        """Optional observations beyond this prefix are omitted without failing the transcript."""
    def accepts_event(self, retained_bytes: int, event_bytes: int) -> bool:
        """Admission precedes host SDK serialization and native JSON extraction."""
    def project(self, report: SessionReportSnapshot) -> SessionReportSnapshot: ...
    def project_event(self, event: object, retained_bytes: int) -> ProjectedReportEvent | None: ...
    def event_record(
        self,
        event: ProjectedReportEvent,
        identity: tuple[str, str, str],
        timestamp: float,
        disabled: bool,
    ) -> LogRecordWire | None: ...

class LoopMonitor:
    """A loop's watchdog: runtime owns thresholds, timing, admission and cleanup, while the host
    supplies asyncio scheduling, stack samples and OpenTelemetry emission.
    """
    def __init__(
        self,
        event_loop: object,
        *,
        warn_threshold: float,
        error_threshold: float,
        tick_interval: float | None,
        name: str,
        emit_spans: bool,
        report: object,
        span_of: object,
    ) -> None: ...
    @property
    def warn_threshold(self) -> float: ...
    @property
    def error_threshold(self) -> float: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

def _core_loop_monitor_thresholds(
    warn_ms: str | None,
    error_ms: str | None,
) -> tuple[float, float] | None: ...

class _core_EventRetention:
    """Bounded retention of a session report's host event snapshots. The native policy retains only
    byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
    """
    def __init__(self, capture_content: bool) -> None: ...
    def claim(self, job_id: str) -> None:
        """Bind report access to the first job that starts the session. Ownership cannot transfer."""
    def allows_read(self, job_id: str) -> bool:
        """Sessions started outside a job retain their existing application-owned access contract."""
    @property
    def capture_content(self) -> bool:
        """Immutable consent captured when this session's report retention was created."""
    def accepts(self, kind: str, redaction: bool) -> bool:
        """Reject content without creation-time consent, unknown schemas and live-object events."""
    def admit(self, retained_bytes: int, now_ns: int) -> tuple[int, bool]:
        """Expire old entries, then admit only within the item and byte budgets. The adapter removes
        exactly the returned prefix before appending the newly admitted snapshot.
        """
    def expire(self, now_ns: int) -> int: ...
    @property
    def diagnostics(self) -> list[tuple[str, int]]: ...

def _core_format_trace_parent(trace_id: int, span_id: int, sampled: bool) -> str | None:
    """Wire values from the SDK's integer identifiers. Invalid parents are omitted."""

def _core_trace_parent(traceparent: str | None = None) -> tuple[int, int, bool] | None:
    """SDK construction values from a validated parent. Invalid context never refuses the
    request.
    """

class _core_SessionReportUpload:
    """Credentials for the session recording endpoint and host OTLP exporters."""
    def __init__(self, api_key: str, api_secret: str) -> None: ...
    def bearer(self) -> str:
        """A fresh LiveKit observability credential. No token policy lives in the host adapter."""
    def content_policy(
        self,
        transcript_requested: bool,
        audio_requested: bool,
        capture_content: bool,
        redaction_enabled: bool,
    ) -> tuple[bool, bool]:
        """Recording options are explicit consent for audio; transcript also requires GenAI content
        capture. Project redaction excludes both raw content formats.
        """
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
        """Returns false only when the project disabled recording. Export failures are reported once
        to the caller, which decides where to surface them without failing the conversation.
        """

class _core_PreparedTranscript:
    """Incremental report preparation. Host adapters admit one item's shape before conversion and
    append it off their event loop; no whole-history extraction crosses a binding call.
    """
    def __init__(self) -> None: ...
    @property
    def input_batch_size(self) -> int:
        """Maximum items per append batch before the adapter returns control to its event loop."""
    def content_policy(self) -> tuple[bool, bool]:
        """Which media parts belong to the transcript's native projection (image, audio)."""
    def check_input(self, items: int, item_bytes: int) -> None:
        """Validate host collection lengths and the shared retained-byte verdict before converting an
        item's fields. These scalar facts cannot allocate an unbounded native item.
        """
    def append(self, item: ChatItem) -> None: ...
    def finish(self) -> str: ...
    def append_event(self, event: object) -> bool:
        """Append one already projected event to the same bounded recording preparation."""
    async def finish_report(self, report: SessionReportSnapshot, path: str) -> None:
        """Serialize and write a console report without converting its history back into host objects."""
