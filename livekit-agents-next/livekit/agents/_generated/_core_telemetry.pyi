# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from livekit.agents._core import SpanKind, StatusCode

from .telemetry import EvaluationRecord, Outcome, ReportTag, SpanContext, TelemetryDiagnostics

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
        kind: SpanKind | None = None,
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
    def set_attribute(self, key: str, value: object) -> None:
        """A JSON scalar or array; `None` removes nothing and is ignored."""
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, object] | None = None,
        timestamp: int | None = None,
    ) -> None: ...
    def set_status(self, code: StatusCode, description: str | None = None) -> None:
        """`description` is kept only with `Error`; `Ok` is final."""
    def update_name(self, name: str) -> None: ...
    def end(self, end_time: int | None = None) -> None:
        """Ends the span; later calls do nothing."""

def _core_telemetry_install(sdk_version: str) -> None:
    """Starts the process's exporter, stamping `sdk_version`, with the application endpoint from
    the `OTEL_EXPORTER_OTLP_*` environment. Later calls do nothing.
    """

def _core_telemetry_configure(
    *,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    allow_pii: bool | None = None,
) -> None:
    """Replaces the application endpoint; arguments override the environment."""

def _core_telemetry_diagnostics() -> TelemetryDiagnostics: ...

def _core_telemetry_conflict() -> None:
    """Counts a configuration conflict the host found (a tracer provider it cannot bridge)."""

def _core_set_capture_content(enabled: bool) -> None:
    """GenAI message capture for spans recorded afterwards."""

async def _core_telemetry_shutdown() -> None:
    """The process-exit flush: every destination's queue, bounded."""

def _core_event_loop_blocked(
    duration: float,
    *,
    parent: SpanContext | None = None,
    job: str | None = None,
    task: str | None = None,
) -> None:
    """A stall of the host's event loop, `duration` seconds long, reported once it ended: its
    measurement, and an `event_loop_blocked` span under `parent`, at most 30 a minute. `task`:
    the task that was running.
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
        verdict: str,
        reasoning: str,
        instructions: str,
    ) -> None:
        """One judgment of a judge group's result: its ``lk.judge.<name>:<verdict>`` tag and record."""

class _core_EventRetention:
    """A session report's host event snapshots: which are accepted and how many are kept. The adapter
    owns the snapshots.
    """
    def __new__(cls) -> Self: ...
    def claim(self, job_id: str) -> None:
        """Bind report access to the first job that starts the session. Ownership cannot transfer."""
    def allows_read(self, job_id: str) -> bool:
        """Sessions started outside a job retain their existing application-owned access contract."""
    def accepts(self, kind: str, redaction: bool) -> bool:
        """Reject content of a redacted session, unknown schemas and live-object events."""
    def admit(self) -> None:
        """Counts one retained event; every accepted event is kept."""

TELEMETRY_MAX_DEPTH: int
