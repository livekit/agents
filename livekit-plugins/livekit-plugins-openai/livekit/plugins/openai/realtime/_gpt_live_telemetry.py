"""The observable Live protocol, without reconstructing managed context or reasoning."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from livekit.agents.telemetry import gen_ai, tracer

from ..log import logger

_TRANSCRIPT_DELTAS = ("session.input_transcript.delta", "session.output_transcript.delta")


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
        self._session_id = ""
        self._output_audio: tuple[int, int, int] | None = None
        self._speech: dict[str, _SpeechSegment] = {}

    def record(
        self, direction: Literal["sent", "received", "send_failed"], event: dict[str, Any]
    ) -> None:
        self._guard(self._record, direction, event)

    def speech(
        self, segment_id: str, role: str, text: str, start_ms: int | None, end_ms: int | None
    ) -> None:
        """Grow one speaker's segment to its text so far, opening its span on the first fragment."""
        self._guard(self._grow_speech, segment_id, role, text, start_ms, end_ms)

    def end_speech(self, segment_id: str) -> None:
        self._guard(self._end_speech, segment_id)

    def close(self) -> None:
        self._guard(self._close)
        self._output_audio = None
        self._speech.clear()
        self._session_id = ""

    def _guard(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:
            # Never let exporter/payload errors change protocol execution or log content.
            logger.warning(
                "gpt-live protocol tracing failed", extra={"error_type": type(exc).__name__}
            )

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
            self._session_id = event.get("session", {}).get("id", "")
        inner = event.get("event", {}) if kind == "response.event" else event
        inner_kind = inner.get("type", kind)
        if kind == "response.event" and inner_kind.endswith(".delta"):
            # Token streams are not lifecycle; the backend response span keeps completed output.
            # Every other nested type is kept, since the service may forward new ones.
            return
        response = inner.get("response")
        if not isinstance(response, dict):
            response = {}
        attrs: dict[str, AttributeValue] = {
            "lk.session_id": self._session_id,
            "lk.openai.direction": direction,
            "lk.openai.event_type": inner_kind,
        }
        for key in (
            "event_id",
            "client_event_id",
            "delegation_id",
            "offset_ms",
            "start_ms",
            "end_ms",
            "sequence_number",
            "reason",
            "code",
            "param",
        ):
            value = event.get(key, inner.get(key))
            if isinstance(value, (str, int, float, bool)):
                attrs[f"lk.openai.{key}"] = value
        for group, source, keys in (
            ("delegation", event.get("delegation"), ("id", "target", "response_id")),
            ("response", response, ("id", "model", "status")),
            ("response.error", response.get("error"), ("code",)),
            ("response.incomplete_details", response.get("incomplete_details"), ("reason",)),
            ("item", inner.get("item"), ("id", "type", "status", "call_id", "name")),
            ("error", event.get("error"), ("type", "code", "param", "client_event_id")),
            ("usage", event.get("usage"), ("seconds",)),
            ("context_window", event.get("context_window"), ("usage_ratio",)),
        ):
            if isinstance(source, dict):
                for key in keys:
                    value = source.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        attrs[f"lk.openai.{group}.{key}"] = value
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
                    item = event.get("item") or {}
                    if item.get("type") == "function_call_output":
                        content["output"] = item.get("output")
                if content:
                    span.set_attribute("lk.pii.openai_content", json.dumps(content))

    def _grow_speech(
        self, segment_id: str, role: str, text: str, start_ms: int | None, end_ms: int | None
    ) -> None:
        now = time.time_ns()
        if (segment := self._speech.get(segment_id)) is None:
            attrs: dict[str, AttributeValue] = {
                "lk.session_id": self._session_id,
                "lk.openai.role": role,
            }
            if start_ms is not None:
                attrs["lk.openai.start_ms"] = start_ms
            span = tracer.start_span(
                "gpt_live.speech", context=self._context, start_time=now, attributes=attrs
            )
            segment = self._speech[segment_id] = _SpeechSegment(span)
        segment.text, segment.end_ms, segment.last_ns = text, end_ms, now
        segment.fragments += 1

    def _end_speech(self, segment_id: str) -> None:
        if (segment := self._speech.pop(segment_id, None)) is None:
            return
        if segment.end_ms is not None:
            segment.span.set_attribute("lk.openai.end_ms", segment.end_ms)
        segment.span.set_attribute("lk.openai.fragment_count", segment.fragments)
        if gen_ai.capture_content_enabled():
            segment.span.set_attribute(
                "lk.pii.openai_content", json.dumps({"transcript": segment.text})
            )
        # A segment closes on the next pause or turn; it spans first to last fragment.
        segment.span.end(end_time=segment.last_ns)

    def _flush_audio(self, count: int, start: int, end: int) -> None:
        span = tracer.start_span(
            "gpt_live.audio_activity",
            context=self._context,
            start_time=start,
            attributes={
                "lk.session_id": self._session_id,
                "lk.openai.event_type": "session.output_audio.delta",
                "lk.openai.direction": "received",
                "lk.openai.frame_count": count,
            },
        )
        span.end(end_time=end)

    def _close(self) -> None:
        if self._session_id:
            self._record("received", {"type": "connection.closed", "reason": "adapter_cleanup"})
        if self._output_audio:
            self._flush_audio(*self._output_audio)
        for segment_id in list(self._speech):
            self._end_speech(segment_id)
