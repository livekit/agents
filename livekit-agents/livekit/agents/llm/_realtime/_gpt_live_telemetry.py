"""The observable Live protocol, without reconstructing managed context or reasoning."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from livekit.agents import llm
from livekit.agents.log import logger
from livekit.agents.telemetry import gen_ai, trace_types, tracer

from . import gpt_live_types as types

_TRANSCRIPT_DELTAS = ("session.input_transcript.delta", "session.output_transcript.delta")


def _guard(message: str, fn: Callable[..., None], *args: Any) -> None:
    try:
        fn(*args)
    except Exception as exc:
        # Never let exporter/payload errors change protocol execution or log content.
        logger.warning(message, extra={"error_type": type(exc).__name__})


def _first_scalar(*values: Any) -> AttributeValue | None:
    for value in values:
        if isinstance(value, (str, int, float, bool)):
            return value
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass
class _SpeechSegment:
    span: trace.Span
    text: str = ""
    end_ms: int | None = None
    fragments: int = 0
    last_ns: int = 0


class ProtocolTrace:
    """Record transport boundaries separately from inference and actual playback.

    Speech is one span per transcript segment. Output audio keeps one bounded counter, not
    payloads or per-frame spans; input audio flows through silence, so it is not traced.
    ``sent`` means the local WebSocket send completed, not that the service accepted it.
    """

    def __init__(self) -> None:
        self._context = trace.set_span_in_context(trace.get_current_span())
        self._started = False
        self._output_audio: tuple[int, int, int] | None = None
        self._speech: dict[str, _SpeechSegment] = {}

    def record(
        self, direction: Literal["sent", "received", "send_failed"], event: dict[str, Any]
    ) -> None:
        _guard("gpt-live protocol tracing failed", self._record, direction, event)

    def speech(
        self, segment_id: str, role: str, text: str, start_ms: int | None, end_ms: int | None
    ) -> None:
        """Grow one speaker's segment to its text so far, opening its span on the first fragment."""
        _guard(
            "gpt-live protocol tracing failed",
            self._grow_speech,
            segment_id,
            role,
            text,
            start_ms,
            end_ms,
        )

    def end_speech(self, segment_id: str) -> None:
        _guard("gpt-live protocol tracing failed", self._end_speech, segment_id)

    def close(self) -> None:
        _guard("gpt-live protocol tracing failed", self._close)
        self._output_audio = None
        self._speech.clear()
        self._started = False

    def _record(self, direction: str, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        if kind == "session.input_audio.append" and direction == "sent":
            return
        if kind == "session.output_audio.delta":
            now = time.time_ns()
            count, start, end = self._output_audio or (0, now, now)
            if now - start >= 1_000_000_000:
                self._flush_audio(count, start, end)
                count, start = 0, now
            self._output_audio = (count + 1, start, now)
            return
        if kind in _TRANSCRIPT_DELTAS:
            return
        if kind == "session.started":
            self._started = True
        inner = _dict(event.get("event")) if kind == "response.event" else event
        inner_kind = inner.get("type", kind)
        if kind == "response.event" and inner_kind.endswith(".delta"):
            # Token streams are not lifecycle; the backend response span keeps completed output.
            # Every other nested type is kept, since the service may forward new ones.
            return
        response = _dict(inner.get("response"))
        delegation = _dict(event.get("delegation"))
        error = _dict(event.get("error"))
        item = _dict(inner.get("item"))

        def top(key: str) -> tuple[Any, Any]:
            return event.get(key), inner.get(key)

        attrs: dict[str, AttributeValue] = {
            trace_types.ATTR_REALTIME_DIRECTION: direction,
            trace_types.ATTR_REALTIME_EVENT_TYPE: inner_kind,
        }
        for key, values in (
            (trace_types.ATTR_REALTIME_EVENT_ID, top("event_id")),
            (
                trace_types.ATTR_REALTIME_CLIENT_EVENT_ID,
                (*top("client_event_id"), error.get("client_event_id")),
            ),
            (trace_types.ATTR_DELEGATION_ID, (*top("delegation_id"), delegation.get("id"))),
            (trace_types.ATTR_REALTIME_SEQUENCE_NUMBER, top("sequence_number")),
            (trace_types.ATTR_REALTIME_OFFSET_MS, top("offset_ms")),
            (trace_types.ATTR_REALTIME_START_MS, top("start_ms")),
            (trace_types.ATTR_REALTIME_END_MS, top("end_ms")),
            (
                trace_types.ATTR_REALTIME_REASON,
                (*top("reason"), _dict(response.get("incomplete_details")).get("reason")),
            ),
            (trace_types.ATTR_REALTIME_ERROR_TYPE, (error.get("type"),)),
            (
                trace_types.ATTR_REALTIME_ERROR_CODE,
                (*top("code"), error.get("code"), _dict(response.get("error")).get("code")),
            ),
            (trace_types.ATTR_REALTIME_ERROR_PARAM, (*top("param"), error.get("param"))),
            (trace_types.ATTR_REALTIME_DELEGATION_TARGET, (delegation.get("target"),)),
            (
                trace_types.ATTR_REALTIME_RESPONSE_ID,
                (response.get("id"), delegation.get("response_id")),
            ),
            (trace_types.ATTR_REALTIME_RESPONSE_MODEL, (response.get("model"),)),
            (trace_types.ATTR_REALTIME_RESPONSE_STATUS, (response.get("status"),)),
            (trace_types.ATTR_REALTIME_ITEM_ID, (item.get("id"),)),
            (trace_types.ATTR_REALTIME_ITEM_TYPE, (item.get("type"),)),
            (trace_types.ATTR_REALTIME_ITEM_STATUS, (item.get("status"),)),
            (trace_types.ATTR_REALTIME_ITEM_CALL_ID, (item.get("call_id"),)),
            (trace_types.ATTR_REALTIME_ITEM_NAME, (item.get("name"),)),
            (trace_types.ATTR_REALTIME_USAGE_SECONDS, (_dict(event.get("usage")).get("seconds"),)),
            (
                trace_types.ATTR_REALTIME_CONTEXT_USAGE_RATIO,
                (_dict(event.get("context_window")).get("usage_ratio"),),
            ),
        ):
            if (value := _first_scalar(*values)) is not None:
                attrs[key] = value
        with tracer.start_as_current_span(
            "gpt_live.protocol", context=self._context, attributes=attrs
        ) as span:
            if direction == "send_failed" or inner_kind in (
                "error",
                "response.failed",
                "response.incomplete",
            ):
                span.set_status(trace.StatusCode.ERROR)
            if gen_ai.capture_content_enabled():
                content: dict[str, Any] = {}
                if kind in (
                    "session.instructions.append",
                    "session.thinking.append",
                    "session.commentary.append",
                ):
                    content["content"] = event.get("content")
                elif kind in ("session.start", "session.update"):
                    config = event.get("session") or {}
                    content = {
                        key: config[key]
                        for key in ("instructions", "input", "delegation")
                        if key in config
                    }
                elif kind == "response.item.create":
                    created = event.get("item") or {}
                    if created.get("type") == "function_call_output":
                        content["output"] = created.get("output")
                if content:
                    span.set_attribute(trace_types.ATTR_REALTIME_EVENT_CONTENT, json.dumps(content))

    def _grow_speech(
        self, segment_id: str, role: str, text: str, start_ms: int | None, end_ms: int | None
    ) -> None:
        now = time.time_ns()
        if (segment := self._speech.get(segment_id)) is None:
            attrs: dict[str, AttributeValue] = {trace_types.ATTR_REALTIME_ROLE: role}
            if start_ms is not None:
                attrs[trace_types.ATTR_REALTIME_START_MS] = start_ms
            span = tracer.start_span(
                "gpt_live.speech", context=self._context, start_time=now, attributes=attrs
            )
            segment = self._speech[segment_id] = _SpeechSegment(span)
        segment.text, segment.end_ms, segment.last_ns = text, end_ms, now
        segment.fragments += 1

    def _end_speech(self, segment_id: str) -> None:
        if (segment := self._speech.pop(segment_id, None)) is None:
            return
        try:
            if segment.end_ms is not None:
                segment.span.set_attribute(trace_types.ATTR_REALTIME_END_MS, segment.end_ms)
            segment.span.set_attribute(trace_types.ATTR_REALTIME_FRAGMENT_COUNT, segment.fragments)
            if gen_ai.capture_content_enabled():
                segment.span.set_attribute(trace_types.ATTR_REALTIME_TRANSCRIPT, segment.text)
        finally:
            # A segment closes on the next pause or turn; it spans first to last fragment.
            segment.span.end(end_time=segment.last_ns)

    def _flush_audio(self, count: int, start: int, end: int) -> None:
        span = tracer.start_span(
            "gpt_live.audio_activity",
            context=self._context,
            start_time=start,
            attributes={
                trace_types.ATTR_REALTIME_EVENT_TYPE: "session.output_audio.delta",
                trace_types.ATTR_REALTIME_DIRECTION: "received",
                trace_types.ATTR_REALTIME_FRAME_COUNT: count,
            },
        )
        span.end(end_time=end)

    def _close(self) -> None:
        if self._started:
            self._record("received", {"type": "connection.closed", "reason": "adapter_cleanup"})
        if self._output_audio:
            self._flush_audio(*self._output_audio)
        for segment_id in list(self._speech):
            self._end_speech(segment_id)


@dataclass
class BackendRequest:
    """What the session configured for the managed backend; the service owns its exact input."""

    provider: str
    model: str
    instructions: str | None
    tools: list[llm.Tool]


@dataclass
class _ResponseTrace:
    """Completed output items for one managed backend response."""

    span: trace.Span
    record_content: bool
    output: list[dict[str, Any]] = field(default_factory=list)
    has_tool_calls: bool = False

    def finish(self, error: str | None = None) -> None:
        """Export the response once its terminal event arrives or the connection ends."""
        try:
            reason = "error" if error else "tool_call" if self.has_tool_calls else "stop"
            gen_ai.set_response_attributes(self.span, finish_reasons=[reason])
            gen_ai.set_content_attributes(self.span, output_messages=self.output)
            if error:
                self.span.set_attribute(trace_types.ATTR_ERROR_TYPE, error)
                self.span.set_status(trace.StatusCode.ERROR)
        finally:
            self.span.end()


class BackendTrace:
    """One ``llm_request`` span per managed backend response, apart from the speaker history.

    A delegation's responses arrive in sequence, so each is keyed by its delegation id (a
    response the application started itself has none). Only completed output items are kept:
    the terminal response snapshot has an empty ``output``.
    """

    def __init__(self, request: Callable[[], BackendRequest]) -> None:
        self._request = request
        self._responses: dict[str | None, _ResponseTrace] = {}

    def response_event(self, envelope: types.ResponseEventEnvelope) -> None:
        _guard("gpt-live backend tracing failed", self._response_event, envelope)

    def function_call(self, delegation_id: str | None, fnc_call: llm.FunctionCall) -> None:
        """Record a call the session accepted, after its validation and deduplication."""
        _guard("gpt-live backend tracing failed", self._function_call, delegation_id, fnc_call)

    def close(self) -> None:
        """End every open response, each guarded, so one failure cannot skip the rest."""
        responses, self._responses = self._responses, {}
        for response in responses.values():
            _guard("gpt-live backend tracing failed", response.finish, "connection_closed")

    def _response_event(self, envelope: types.ResponseEventEnvelope) -> None:
        event, d_id = envelope.event, envelope.delegation_id
        if event.type == "response.created":
            if previous := self._responses.pop(d_id, None):
                previous.finish("response_replaced")
            self._responses[d_id] = self._start(d_id)
        if (current := self._responses.get(d_id)) is None:
            return
        if (response := event.response) is not None:
            gen_ai.set_response_attributes(
                current.span, response_id=response.id, model=response.model
            )
            if usage := response.usage:
                gen_ai.set_usage_attributes(
                    current.span,
                    llm.CompletionUsage(
                        prompt_tokens=usage.input_tokens,
                        prompt_cached_tokens=usage.input_tokens_details.cached_tokens,
                        cache_creation_tokens=usage.input_tokens_details.cache_write_tokens,
                        completion_tokens=usage.output_tokens,
                        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
                        total_tokens=usage.total_tokens,
                    ),
                )
        if (
            event.type == "response.output_item.done"
            and (item := event.item) is not None
            and item.status == "completed"
            and item.type == "message"
            and current.record_content
        ):
            parts = [
                {"type": "text", "content": part.text or part.refusal}
                for part in item.content
                if part.type in ("output_text", "refusal") and (part.text or part.refusal)
            ]
            if parts:
                current.output.append({"role": "assistant", "parts": parts})
        if event.type in ("response.completed", "response.failed", "response.incomplete"):
            self._responses.pop(d_id)
            failure = None
            if event.type == "response.failed":
                failure = ((response and response.error) or {}).get("code") or event.type
            elif event.type == "response.incomplete":
                details = (response and response.incomplete_details) or {}
                failure = details.get("reason") or event.type
            current.finish(failure)

    def _start(self, delegation_id: str | None) -> _ResponseTrace:
        request = self._request()
        span = tracer.start_span("llm_request", kind=trace.SpanKind.CLIENT)
        response = _ResponseTrace(span, gen_ai.capture_content_enabled() and span.is_recording())
        gen_ai.set_request_attributes(
            span, operation="chat", provider=request.provider, model=request.model, stream=True
        )
        if delegation_id is not None:
            span.set_attribute(trace_types.ATTR_DELEGATION_ID, delegation_id)
        if response.record_content:
            gen_ai.set_content_attributes(
                span,
                system_instructions=[{"type": "text", "content": request.instructions}]
                if request.instructions
                else None,
                tool_definitions=gen_ai.to_tool_definitions(request.tools),
            )
        return response

    def _function_call(self, delegation_id: str | None, fnc_call: llm.FunctionCall) -> None:
        if (current := self._responses.get(delegation_id)) is None:
            return
        current.has_tool_calls = True
        if current.record_content:
            current.output.extend(gen_ai.to_output_messages(function_calls=[fnc_call]))
