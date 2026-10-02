"""The observable Live protocol, without reconstructing managed context or reasoning."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from livekit.agents.telemetry import gen_ai, tracer

from ..log import logger


class ProtocolTrace:
    """Record transport boundaries separately from inference and actual playback.

    Audio is continuous, including silence: activity does not prove speech. Keep
    one bounded counter per direction, not audio payloads or per-frame spans.
    """

    def __init__(self) -> None:
        self._context = trace.set_span_in_context(trace.get_current_span())
        self._session_id = ""
        self._audio: dict[str, tuple[int, int, int]] = {}

    def record(
        self, direction: Literal["queued", "sent", "received", "send_failed"], event: dict[str, Any]
    ) -> None:
        try:
            self._record(direction, event)
        except Exception as exc:
            # Never let exporter/payload errors change protocol execution or log content.
            logger.warning(
                "gpt-live protocol tracing failed", extra={"error_type": type(exc).__name__}
            )

    def _record(self, direction: str, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        now = time.time_ns()
        if (
            kind in ("session.input_audio.append", "session.output_audio.delta")
            and direction != "send_failed"
        ):
            if direction not in ("sent", "received"):
                return
            count, start, end = self._audio.get(kind, (0, now, now))
            if now - start >= 1_000_000_000:
                self._flush_audio(kind, count, start, end)
                count, start = 0, now
            self._audio[kind] = (count + 1, start, now)
            return
        if kind == "session.started":
            self._session_id = event.get("session", {}).get("id", "")
        inner = event.get("event", {}) if kind == "response.event" else event
        inner_kind = inner.get("type", kind)
        if kind == "response.event" and inner_kind not in (
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
            "response.output_item.done",
        ):
            return  # Completed items are captured by the backend response span.
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
        ):
            value = event.get(key, inner.get(key))
            if isinstance(value, (str, int, float, bool)):
                attrs[f"lk.openai.{key}"] = value
        for group, source, keys in (
            ("delegation", event.get("delegation"), ("id", "target", "response_id")),
            ("response", inner.get("response"), ("id", "model", "status")),
            ("item", inner.get("item"), ("id", "type", "status", "call_id", "name")),
            ("error", event.get("error"), ("type", "code", "client_event_id")),
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
            # One payload copy at submission/receipt; sent is only transport proof.
            if gen_ai.capture_content_enabled() and direction in ("queued", "received"):
                content: dict[str, Any] = {}
                if kind in ("session.input_transcript.delta", "session.output_transcript.delta"):
                    content["delta"] = event.get("delta")
                elif kind in (
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

    def _flush_audio(self, kind: str, count: int, start: int, end: int) -> None:
        span = tracer.start_span(
            "gpt_live.audio_activity",
            context=self._context,
            start_time=start,
            attributes={
                "lk.session_id": self._session_id,
                "lk.openai.event_type": kind,
                "lk.openai.direction": "sent"
                if kind == "session.input_audio.append"
                else "received",
                "lk.openai.frame_count": count,
            },
        )
        span.end(end_time=end)

    def close(self) -> None:
        try:
            if self._session_id:
                self.record("received", {"type": "connection.closed", "reason": "adapter_cleanup"})
            for kind, (count, start, end) in self._audio.items():
                self._flush_audio(kind, count, start, end)
        except Exception as exc:
            logger.warning(
                "gpt-live protocol tracing failed", extra={"error_type": type(exc).__name__}
            )
        finally:
            self._audio.clear()
            self._session_id = ""
