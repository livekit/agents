"""Data models for the post-call telemetry report.

Beta: this schema is not covered by semver stability guarantees and may change in a
future release.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import json
import math
import time
import uuid
from typing import (  # noqa: UP035 - Dict/List/Union needed for the recursive alias below
    Any,
    Dict,
    List,
    Literal,
    Union,
)

from pydantic import BaseModel, Field
from typing_extensions import TypeAliasType

from ...llm import ChatContext, ChatMessage, ChatRole, FunctionCall
from ...metrics import ModelUsage
from ...version import __version__
from ...voice.events import (
    AgentEvent,
    CloseEvent,
    FunctionToolsExecutedEvent,
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
)

# A plain recursive `TypeAlias` union causes pydantic's schema generator to recurse
# infinitely; `TypeAliasType` (PEP 695 backport) is pydantic's documented way to define
# a recursive JSON-safe-value type — it builds a ref-based schema instead of inlining.
# mypy (2.3.0, checked in isolation without this repo's config too) cannot resolve the
# self-reference through an explicit `TypeAliasType(...)` call — only through the native
# `type X = ...` statement, which is Python 3.12+ only and unusable at this repo's
# py310 floor — so this single-line definition is ignored; pydantic itself builds and
# uses the schema correctly at runtime (see tests/test_gtm_telemetry_models.py).
JsonValue = TypeAliasType("JsonValue", Union[Dict[str, "JsonValue"], List["JsonValue"], str, int, float, bool, None])  # type: ignore[misc]  # noqa: UP006, UP007 - Dict/List/Union needed, see above  # fmt: skip
"""A JSON-safe value: the output of :func:`to_json_safe`."""

_MAX_DEPTH = 20


def to_json_safe(value: Any, *, _depth: int = 0, _seen: frozenset[int] = frozenset()) -> JsonValue:
    """Recursively normalize an arbitrary value into JSON-safe primitives.

    None/bool/int/str pass through unchanged. NaN/Infinity floats become sentinel
    strings (JSON has no representation for them). bytes are base64-encoded. Enums use
    their ``.value``. Dataclasses and pydantic models are recursed into field-by-field.
    dict keys are coerced to ``str``. list/tuple become JSON arrays; set/frozenset
    become a deterministically sorted JSON array. Exceptions become
    ``{"type": ..., "message": ...}`` — never a traceback. Anything else falls back to
    ``f"<unserializable:{type(value).__name__}>"`` — never ``repr()``/``str()`` of an
    arbitrary object, since a default object repr can leak a memory address or a
    private attribute. Guards against cycles (by ``id()``) and excessive depth.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "<nan>"
        if math.isinf(value):
            return "<inf>" if value > 0 else "<-inf>"
        return value

    if _depth >= _MAX_DEPTH:
        return "<max-depth-exceeded>"

    if isinstance(value, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii")}

    if isinstance(value, enum.Enum):
        return to_json_safe(value.value, _depth=_depth + 1, _seen=_seen)

    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}

    if isinstance(value, BaseModel):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular>"
        return to_json_safe(
            value.model_dump(mode="json"), _depth=_depth + 1, _seen=_seen | {obj_id}
        )

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular>"
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return to_json_safe(fields, _depth=_depth + 1, _seen=_seen | {obj_id})

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular>"
        seen = _seen | {obj_id}
        return {str(k): to_json_safe(v, _depth=_depth + 1, _seen=seen) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        obj_id = id(value)
        if obj_id in _seen:
            return "<circular>"
        seen = _seen | {obj_id}
        items = [to_json_safe(v, _depth=_depth + 1, _seen=seen) for v in value]
        if isinstance(value, (set, frozenset)):
            # sets have no stable order; sort the normalized items for determinism
            items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items

    return f"<unserializable:{type(value).__name__}>"


def _parse_json_ish(raw: str) -> JsonValue:
    """Parse a JSON string when possible; otherwise keep the raw string as-is."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    return to_json_safe(parsed)


class MetricAggregate(BaseModel):
    """A count/sum/min/max/mean aggregate over repeated latency samples."""

    count: int
    sum: float
    min: float
    max: float
    mean: float


class ToolProgressUpdate(BaseModel):
    """A progress update emitted via ``ctx.update()`` while a tool call runs."""

    message: str
    created_at: float


class ToolExecutionRecord(BaseModel):
    """One logical record per function-call ``call_id``.

    ``status`` mirrors ``ToolCallEnded.status`` (``"done"``/``"error"``/``"cancelled"``)
    plus two bookend states: ``"started"`` (a start event was seen but no terminal event
    yet) and ``"interrupted"`` (still ``"started"`` when the session ended — the call was
    cut off before a terminal event arrived).

    Note: this does not attempt to represent whether the *spoken reply* summarizing a
    tool's result was itself interrupted or skipped (``ToolReplyUpdated`` in the
    session's event stream) — that describes reply delivery, not the tool's own
    execution outcome, so it is out of scope for this record.
    """

    call_id: str
    name: str
    arguments: JsonValue = None
    result: JsonValue = None
    status: Literal["started", "done", "error", "cancelled", "interrupted"] = "started"
    is_error: bool = False
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    """Error message when ``status == "error"``. Never a traceback."""
    progress_updates: list[ToolProgressUpdate] = Field(default_factory=list)


class TranscriptTurn(BaseModel):
    """A single conversational turn extracted from the session's chat history."""

    item_id: str
    role: ChatRole
    text: str
    interrupted: bool
    created_at: float


class SessionMetricsSummary(BaseModel):
    """A compact, post-call-analytics-oriented summary of session metrics."""

    model_usage: list[ModelUsage] = Field(default_factory=list)
    """Per-model/provider usage totals, from the session's ``ModelUsageCollector``."""
    latency: dict[str, MetricAggregate] = Field(default_factory=dict)
    """Aggregated per-turn latency samples pulled from ``ChatMessage.metrics``. Keys:
    ``llm_node_ttft``, ``tts_node_ttfb``, ``end_of_turn_delay``, ``transcription_delay``.
    A key is present only if at least one turn carried a sample for it."""
    tool_call_count: int = 0
    tool_error_count: int = 0


class CollectorConfig(BaseModel):
    """Configuration controlling what a :class:`PostCallTelemetryCollector` includes."""

    include_system_messages: bool = False
    """Include system/developer chat turns in the transcript (default: user+assistant only)."""
    include_tool_arguments: bool = True
    """Include parsed tool-call arguments in each :class:`ToolExecutionRecord`."""
    include_tool_results: bool = True
    """Include parsed tool-call results in each :class:`ToolExecutionRecord`."""


class PostCallReport(BaseModel):
    """A deterministic, JSON-serializable post-call report.

    Beta: this schema is not covered by semver stability guarantees.
    """

    schema_version: str = "1"
    report_id: str
    """Generated fresh each time a report is built; stable within one report instance."""
    job_id: str | None = None
    room_id: str | None = None
    room_name: str | None = None
    participant_identity: str | None = None

    started_at: float | None = None
    ended: bool = False
    """True once the session has fully closed. A report built while the session is
    still running is a preliminary, honestly-partial snapshot (``ended=False``)."""
    end_reason: (
        Literal[
            "error", "job_shutdown", "participant_disconnected", "user_initiated", "task_completed"
        ]
        | None
    ) = None
    duration: float | None = None
    """Always >= 0. Computed from the session's own ``close`` event timestamp when
    ``ended``, or "so far" from ``time.time()`` otherwise."""

    transcript: list[TranscriptTurn] = Field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = Field(default_factory=list)
    metrics: SessionMetricsSummary = Field(default_factory=SessionMetricsSummary)

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """User-supplied business metadata, normalized to JSON-safe values."""

    report_created_at: float
    sdk_version: str = __version__

    @classmethod
    def from_session(
        cls,
        *,
        job_id: str | None,
        room_id: str | None,
        room_name: str | None,
        participant_identity: str | None,
        started_at: float | None,
        events: list[AgentEvent],
        chat_history: ChatContext,
        model_usage: list[ModelUsage],
        config: CollectorConfig,
        metadata: dict[str, Any],
    ) -> PostCallReport:
        """Build a report from a single, already-captured snapshot of session state.

        ``events`` should be a snapshot (e.g. ``list(session._recorded_events)``), not a
        live reference, so this reduction sees a consistent, unchanging view.

        ``ended`` is derived solely from whether a ``close`` event is present in
        ``events`` — not from ``AgentSession._started`` — since a session that was
        attached but never started also has ``_started is False``, which would
        otherwise be indistinguishable from "ran and then closed".
        """
        close_event = _find_close_event(events)
        ended = close_event is not None
        end_reason = close_event.reason.value if close_event else None

        now = time.time()
        duration: float | None
        if started_at is None:
            duration = None
        elif close_event is not None:
            duration = max(0.0, close_event.created_at - started_at)
        else:
            duration = max(0.0, now - started_at)

        tool_executions = _build_tool_executions(
            events,
            include_arguments=config.include_tool_arguments,
            include_results=config.include_tool_results,
        )

        return cls(
            report_id=uuid.uuid4().hex,
            job_id=job_id,
            room_id=room_id,
            room_name=room_name,
            participant_identity=participant_identity,
            started_at=started_at,
            ended=ended,
            end_reason=end_reason,
            duration=duration,
            transcript=_build_transcript(
                chat_history, include_system_messages=config.include_system_messages
            ),
            tool_executions=tool_executions,
            metrics=_build_metrics_summary(chat_history, model_usage, tool_executions),
            metadata={str(k): to_json_safe(v) for k, v in metadata.items()},
            report_created_at=now,
            sdk_version=__version__,
        )


def _find_close_event(events: list[AgentEvent]) -> CloseEvent | None:
    for event in reversed(events):
        if isinstance(event, CloseEvent):
            return event
    return None


def _build_transcript(
    chat_history: ChatContext, *, include_system_messages: bool
) -> list[TranscriptTurn]:
    allowed_roles: tuple[ChatRole, ...] = (
        ("user", "assistant", "system", "developer")
        if include_system_messages
        else ("user", "assistant")
    )
    turns: list[TranscriptTurn] = []
    for item in chat_history.items:
        if not isinstance(item, ChatMessage) or item.role not in allowed_roles:
            continue
        turns.append(
            TranscriptTurn(
                item_id=item.id,
                role=item.role,
                text=item.text_content or "",
                interrupted=item.interrupted,
                created_at=item.created_at,
            )
        )
    return turns


def _build_tool_executions(
    events: list[AgentEvent],
    *,
    include_arguments: bool,
    include_results: bool,
) -> list[ToolExecutionRecord]:
    records: dict[str, ToolExecutionRecord] = {}
    order: list[str] = []

    def _get(call_id: str) -> ToolExecutionRecord:
        if call_id not in records:
            records[call_id] = ToolExecutionRecord(call_id=call_id, name="")
            order.append(call_id)
        return records[call_id]

    def _set_arguments(record: ToolExecutionRecord, call: FunctionCall) -> None:
        if include_arguments and record.arguments is None:
            record.arguments = _parse_json_ish(call.arguments)

    for event in events:
        if isinstance(event, ToolExecutionUpdatedEvent):
            update = event.update
            if isinstance(update, ToolCallStarted):
                record = _get(update.function_call.call_id)
                record.name = update.function_call.name
                record.started_at = update.function_call.created_at
                _set_arguments(record, update.function_call)
            elif isinstance(update, ToolCallUpdated):
                record = _get(update.call_id)
                record.progress_updates.append(
                    ToolProgressUpdate(message=update.message, created_at=event.created_at)
                )
            elif isinstance(update, ToolCallEnded):
                record = _get(update.call_id)
                record.ended_at = event.created_at
                if update.status == "error":
                    record.status = "error"
                    record.is_error = True
                    record.error = update.message
                else:
                    record.status = update.status  # "done" or "cancelled"
            # ToolReplyUpdated: deliberately not consumed here — see ToolExecutionRecord
            # docstring for why.
        elif isinstance(event, FunctionToolsExecutedEvent):
            for call, output in event.zipped():
                record = _get(call.call_id)
                if not record.name:
                    record.name = call.name
                if record.started_at is None:
                    record.started_at = call.created_at
                _set_arguments(record, call)
                if output is None:
                    continue
                if record.ended_at is None:
                    record.ended_at = output.created_at
                if include_results:
                    record.result = _parse_json_ish(output.output)
                if output.is_error:
                    record.is_error = True
                    if record.status == "started":
                        record.status = "error"
                    if record.error is None:
                        record.error = output.output

    for call_id in order:
        record = records[call_id]
        if record.status == "started":
            record.status = "interrupted"

    return [records[call_id] for call_id in order]


def _build_metrics_summary(
    chat_history: ChatContext,
    model_usage: list[ModelUsage],
    tool_executions: list[ToolExecutionRecord],
) -> SessionMetricsSummary:
    latency_keys = ("llm_node_ttft", "tts_node_ttfb", "end_of_turn_delay", "transcription_delay")
    samples: dict[str, list[float]] = {key: [] for key in latency_keys}

    for item in chat_history.items:
        if not isinstance(item, ChatMessage):
            continue
        for key in latency_keys:
            value = item.metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                samples[key].append(float(value))

    latency: dict[str, MetricAggregate] = {}
    for key, values in samples.items():
        if not values:
            continue
        latency[key] = MetricAggregate(
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=sum(values) / len(values),
        )

    return SessionMetricsSummary(
        model_usage=list(model_usage),
        latency=latency,
        tool_call_count=len(tool_executions),
        tool_error_count=sum(1 for record in tool_executions if record.is_error),
    )
