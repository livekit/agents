# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from opentelemetry.trace import SpanKind as HostSpanKind, StatusCode as HostStatusCode
from opentelemetry.util.types import AttributeValue

from livekit.agents._core import SpanKind, StatusCode

from .telemetry import (
    EndedSpan,
    EvaluationRecord,
    Outcome,
    ReportTag,
    SessionReport,
    SpanContext,
    TelemetryDiagnostics,
)

class TelemetrySpan:
    """A span started through the host's OpenTelemetry API (an application span, or a framework span
    around host-only code). Ended once; recorded by the process's exporter.
    """
    def __new__(
        cls,
        name: str,
        scope: str,
        *,
        parent: SpanContext | None = None,
        kind: SpanKind | HostSpanKind | None = None,
        start_time: int | None = None,
        ids: SpanContext | None = None,
        job: str | None = None,
    ) -> Self:
        """`parent`: the current span, `None` for a new trace. `ids`: a span the host already started (an SDK span the bridge forwards). `job`: the
        job it belongs to.
        """
    @property
    def context(self) -> SpanContext: ...
    def is_recording(self) -> bool:
        """Whether the span has not ended."""
    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """`None` removes nothing and is ignored."""
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] | None = None,
        timestamp: int | None = None,
    ) -> None: ...
    def set_status(self, code: StatusCode | HostStatusCode, description: str | None = None) -> None:
        """`description` is kept only with `Error`; `Ok` is final."""
    def update_name(self, name: str) -> None: ...
    def end(self, end_time: int | None = None) -> None:
        """Ends the span; later calls do nothing."""

def _core_telemetry_install(sdk_version: str) -> None:
    """Starts the process's exporter, stamping `sdk_version`, with the application endpoint from
    the `OTEL_EXPORTER_OTLP_*` environment. Later calls do nothing.
    """

def configure(
    *,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    allow_pii: bool | None = None,
) -> None:
    """Send the process's spans and metrics to an OTLP/HTTP protobuf endpoint; arguments override
    the `OTEL_EXPORTER_OTLP_*` environment.
    """

def diagnostics() -> TelemetryDiagnostics:
    """The process's export counters."""

def _core_telemetry_conflict() -> None:
    """Counts a configuration conflict the host found (a tracer provider it cannot bridge)."""

def _core_telemetry_export(span: EndedSpan, *, job: str | None = None) -> None:
    """Records a span the application's OpenTelemetry SDK ended (the bridge of an SDK provider),
    counted on `job`'s sessions.
    """

def set_capture_content(enabled: bool) -> None:
    """Whether model request and response messages and judge explanations are recorded."""

async def _core_telemetry_shutdown() -> None:
    """The process-exit flush: every destination's queue, bounded."""

class _core_LoopWatch:
    """One asyncio loop's stall watch (docs/internals/telemetry.md "Event-loop stalls"): the loop's
    own timer beats it every 20 ms; the process's detector records a beat 100 ms old as a stall,
    with the loop thread's stack where it can be read. Beats take no lock.
    """
    def __new__(
        cls,
        os_thread: int,
        *,
        parent: SpanContext | None = None,
        job: str | None = None,
        whole_process: bool = False,
    ) -> Self:
        """Watches the loop running on OS thread `os_thread` (`threading.get_native_id()`); its
        stalls go under `parent` and count on `job`'s sessions. `whole_process`: the process runs
        this job alone (the process executor), so its other running Python threads join the job's
        profile; otherwise the profile holds the loop thread alone.
        """
    def beat(self) -> None:
        """The loop ran."""
    def close(self) -> None:
        """The loop is done; a stall in progress is recorded."""

def session_report_json(report: SessionReport) -> str:
    """The console's ``session_report.json``: the report's identity, events, options, timestamp,
    audio recording, chat history (with timestamps) and usage (without absent fields), as
    Python's ``json.dumps`` writes it.
    """

class Tagger:
    """Tag sessions with metadata for observability.

    The Tagger allows adding string tags (key:value format) with optional
    structured metadata to sessions. Tags and metadata are uploaded to
    LiveKit Cloud at session end.

    Example:
    ```python
    # Mark session as successful
    ctx.tagger.success(reason="Task completed successfully")

    # Mark session as failed
    ctx.tagger.fail(reason="User hung up before completing booking")

    # Add custom tags
    ctx.tagger.add("voicemail:true")
    ctx.tagger.add("language:es")

    # Add tags with structured metadata
    ctx.tagger.add(
        "appointment:booked",
        metadata={"slot_id": "abc123", "calendar": "cal.com"},
    )

    # Remove a tag
    ctx.tagger.remove("voicemail:true")
    ```
    """
    def __new__(cls) -> Self: ...
    def success(self, reason: str | None = None) -> None:
        """Mark the session as successful; ``reason`` is kept apart from the tag."""
    def fail(self, reason: str | None = None) -> None:
        """Mark the session as failed; ``reason`` is kept apart from the tag."""
    def add(self, tag: str, *, metadata: object | None = None) -> None:
        """Add a ``key:value`` tag, with optional structured metadata."""
    def remove(self, tag: str) -> None:
        """Remove a tag; removing an absent one does nothing."""
    @property
    def tags(self) -> frozenset[str]:
        """The session's tags."""
    def entries(self) -> list[ReportTag]:
        """The tags in insertion order, with their metadata and when they were added."""
    @property
    def evaluations(self) -> list[EvaluationRecord]:
        """The judgments recorded by judge groups."""
    @property
    def outcome(self) -> Outcome | None:
        """``"success"``, ``"fail"`` or ``None``."""
    @property
    def outcome_reason(self) -> str | None:
        """The reason given with the outcome."""
    def _core_record_judgment(
        self,
        name: str,
        verdict: Literal["pass", "fail", "maybe"],
        reasoning: str,
        instructions: str,
    ) -> None:
        """One judgment of a judge group's result: its ``lk.judge.<name>:<verdict>`` tag and record."""
    def _core_stamp(self, report: SessionReport) -> SessionReport:
        """``report`` with the session's tags, evaluations and outcome as of now."""
