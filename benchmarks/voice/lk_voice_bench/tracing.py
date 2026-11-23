"""Capture the SDK's OpenTelemetry spans in-process, per session.

Benchmark sessions run without a job, so nothing exports their traces. With ``LK_BENCH_TRACE=1``
the harness installs a tracer provider that keeps finished spans in memory, grouped by trace id;
``BenchAgent`` then writes its own session's spans (end-of-turn decisions with the EOT model's
probability and threshold, user/agent turns, LLM/TTS nodes) into its agent log.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

TRACE_ENV = "LK_BENCH_TRACE"


class _GroupingExporter(SpanExporter):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_trace: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                ctx = span.get_span_context()
                if ctx is None:
                    continue
                self._by_trace[ctx.trace_id].append(
                    {
                        "name": span.name,
                        "start": (span.start_time or 0) / 1e9,
                        "end": (span.end_time or 0) / 1e9,
                        "span_id": format(ctx.span_id, "016x"),
                        "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
                        "attributes": dict(span.attributes or {}),
                    }
                )
        return SpanExportResult.SUCCESS

    def pop(self, trace_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self._by_trace.pop(trace_id, []), key=lambda s: s["start"])

    def shutdown(self) -> None:
        pass


_exporter: _GroupingExporter | None = None
_install_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get(TRACE_ENV, "") not in ("", "0", "false")


def install() -> _GroupingExporter:
    """Install the in-memory tracer provider once per process."""
    global _exporter
    with _install_lock:
        if _exporter is None:
            from livekit.agents.telemetry import set_tracer_provider

            _exporter = _GroupingExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(_exporter))
            set_tracer_provider(provider)
        return _exporter


def pop_trace(trace_id: int) -> list[dict[str, Any]]:
    return _exporter.pop(trace_id) if _exporter is not None else []
