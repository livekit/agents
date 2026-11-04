# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from .llm import ChatChunk
from .metrics import LLMMetrics, STTMetrics, TTSMetrics
from .stt import SpeechEvent
from .telemetry import EvaluationRecord, Outcome, ReportTag
from .tts import SynthesizedAudio

def stt_batch_metrics(
    event: SpeechEvent,
    start: float,
    now: float,
    audio_durations: Sequence[float],
    host: tuple[float, str, bool, str, str],
) -> STTMetrics:
    """`STT.recognize`'s metrics for one batch recognition."""

def stt_stream_metrics(
    event: SpeechEvent,
    host: tuple[float, str, bool, str, str],
) -> tuple[STTMetrics | None, bool]:
    """`RecognizeStream`'s metrics for one event, and whether it resets the retry count."""

class LlmMetricState:
    """`LLMStream`'s metric fold as the host drives it."""
    def __new__(cls, start: float) -> Self: ...
    def observe(self, now: float, chunk: ChatChunk) -> bool:
        """Whether `chunk` is the first with a response."""
    def finish(
        self,
        now: float,
        failed: bool,
        host: tuple[float, str, bool, str, str],
    ) -> LLMMetrics | None:
        """`None` for a failed attempt or one that yielded no chunk."""

class TtsMetricState:
    """The TTS streams' metric fold as the host drives it."""
    def __new__(cls, streamed: bool) -> Self: ...
    def queue_text(self, text: str) -> None:
        """A streamed segment's text, in the order its audio is finished."""
    def observe(self, now: float, start: float, audio: SynthesizedAudio) -> bool:
        """Whether `audio` ends its segment."""
    def finish(
        self,
        now: float,
        start: float,
        failed: bool,
        batch_text: str,
        host: tuple[float, str, bool, str, str],
    ) -> TTSMetrics | None:
        """The record of a finished request or segment, then the figures reset."""

class TelemetrySpan:
    """A span started through the host's OpenTelemetry API (an application span, or a framework span
    around host-only code). Ended once; recorded by the process's exporter.
    """
    def __new__(
        cls,
        name: str,
        scope: str,
        *,
        parent: tuple[int, int, bool] | None = None,
        kind: int = 0,
        start_time: int | None = None,
        ids: tuple[int, int] | None = None,
        job: str | None = None,
    ) -> Self:
        """`parent`: `(trace_id, span_id, sampled)` of the current span, `None` for a new trace.
        `kind`: the OpenTelemetry `SpanKind` value. `ids`: `(trace_id, span_id)` of a span the
        host already started (an SDK span the bridge forwards). `job`: the job it belongs to.
        """
    @property
    def context(self) -> tuple[int, int, bool]:
        """`(trace_id, span_id, sampled)`."""
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
    def set_status(self, code: int, description: str | None = None) -> None:
        """`code`: the OpenTelemetry `StatusCode` value."""
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

def _core_telemetry_diagnostics() -> dict[str, int]:
    """The export counters: `dropped`, `failed`, `conflicts`, `incomplete_cleanup`, `swept_spans`,
    `abandoned_sessions`.
    """

def _core_telemetry_conflict() -> None:
    """Counts a configuration conflict the host found (a tracer provider it cannot bridge)."""

def _core_set_capture_content(enabled: bool) -> None:
    """GenAI message capture for spans recorded afterwards."""

async def _core_telemetry_shutdown() -> None:
    """The process-exit flush: every destination's queue, bounded."""

def _core_event_loop_blocked(
    duration: float,
    *,
    parent: tuple[int, int, bool] | None = None,
    job: str | None = None,
    task: str | None = None,
) -> None:
    """A stall of the host's event loop, `duration` seconds long, reported once it ended: its
    measurement, and an `event_loop_blocked` span under `parent` (`(trace_id, span_id,
    sampled)`), at most 30 a minute. `task`: the task that was running.
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
    def __new__(cls) -> Self:
        """Takes the process's GenAI message capture setting as it is now."""
    def claim(self, job_id: str) -> None:
        """Bind report access to the first job that starts the session. Ownership cannot transfer."""
    def allows_read(self, job_id: str) -> bool:
        """Sessions started outside a job retain their existing application-owned access contract."""
    def accepts(self, kind: str, redaction: bool) -> bool:
        """Reject content without creation-time consent, unknown schemas and live-object events."""
    def admit(self) -> None:
        """Counts one retained event; every accepted event is kept."""

TELEMETRY_MAX_DEPTH: int
