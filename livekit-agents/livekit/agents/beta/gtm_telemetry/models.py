"""Pydantic wire models for the post-call GTM/CRM telemetry report.

All timestamps are float epoch seconds (``time.time()``), matching the repo-wide
event convention (``created_at: float = Field(default_factory=time.time)``).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...utils import shortuuid


class ToolInvocationRecord(BaseModel):
    """A single function-tool invocation captured during the session.

    Timing is self-measured from the ``tool_execution_updated`` lifecycle events
    (no duration field exists on the events themselves). Calls that never reach
    the executor (unknown tools, malformed arguments, rejected duplicates) are
    recorded untimed: ``completed_at`` and ``duration_ms`` stay ``None``.
    """

    tool_name: str
    call_id: str
    arguments: dict[str, Any]
    """Arguments parsed from the ``FunctionCall.arguments`` JSON string.
    Falls back to ``{"_raw": <string>}`` when the string is not valid JSON."""
    result: str | None = None
    """Tool result text (``ToolCallEnded.message`` when the call succeeded,
    or the ``FunctionCallOutput.output`` backfill from the batch event)."""
    error: str | None = None
    """Diagnostic error text. Sourced from ``ToolCallEnded.message`` (the actual
    exception text) in preference to the LLM-facing redacted output."""
    status: Literal["done", "error", "cancelled", "running"] = "running"
    started_at: float
    """Epoch seconds. Lifecycle path: the started-event ``created_at``;
    batch-only records: ``FunctionCall.created_at``."""
    completed_at: float | None = None
    duration_ms: float | None = None
    """Self-timed latency in milliseconds; ``None`` for calls that never
    reached the executor."""


class TranscriptTurn(BaseModel):
    """One committed conversation turn (user or agent)."""

    speaker: Literal["user", "agent"]
    text: str
    interrupted: bool = False
    timestamp: float
    """Epoch seconds (``ChatMessage.created_at``)."""


class CallMetrics(BaseModel):
    """Aggregated call-level metrics.

    Speech durations are ``None`` (not ``0.0``) when no STT/TTS data was
    observed at all — realtime sessions route usage into LLM usage, so a hard
    zero would be indistinguishable from silence.
    """

    total_duration_seconds: float = 0.0
    user_speech_duration_seconds: float | None = None
    """Sum of STT pushed-audio duration in seconds (includes silence pushed to
    the STT); ``None`` when no STT data was observed (e.g. realtime path)."""
    agent_speech_duration_seconds: float | None = None
    """Sum of TTS synthesized-audio duration in seconds; ``None`` when no TTS
    data was observed."""
    total_tool_calls: int = 0
    failed_tool_calls: int = 0
    avg_llm_ttft_ms: float | None = None
    """Mean LLM time-to-first-token in milliseconds; ``None`` when no LLM
    requests were observed."""


class PostCallReport(BaseModel):
    """The complete post-call payload shipped to GTM/CRM webhooks."""

    type: Literal["post_call_report"] = "post_call_report"
    report_id: str = Field(default_factory=lambda: shortuuid("report_"))
    room_name: str | None = None
    room_id: str | None = None
    job_id: str | None = None
    participant_identity: str | None = None
    close_reason: str | None = None
    """``CloseReason.value`` from the session's ``CloseEvent``, when observed."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """User-supplied correlation ids (e.g. CRM contact/deal ids, campaign tags)."""
    turns: list[TranscriptTurn] = Field(default_factory=list)
    tool_invocations: list[ToolInvocationRecord] = Field(default_factory=list)
    metrics: CallMetrics = Field(default_factory=CallMetrics)
    created_at: float = Field(default_factory=time.time)
