from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from opentelemetry import metrics as metrics_api

from .. import _core

if TYPE_CHECKING:
    from ..llm.chat_context import ChatContext

_meter = metrics_api.get_meter("livekit-agents")

_event_loop_blocked = _meter.create_histogram(
    "lk.agents.event_loop.blocked_duration",
    unit="s",
    description="Duration of synchronous blocks detected on an agent event loop",
)


@functools.cache
def _histogram(name: str) -> Any:
    return _meter.create_histogram(name, unit="s")


def _job_attrs() -> dict[str, Any]:
    """The per-job attributes stamped on every measurement: the meter provider outlives jobs, so
    they cannot sit on its resource."""
    from ..job import get_job_context  # local import: job.py imports this module

    ctx = get_job_context(required=False)
    if ctx is None:
        return {}
    if (state := ctx._telemetry_state) is not None:
        return dict(state.attributes)
    # recording was not initialized (disabled, or the crash path); keep identity
    return {"room_id": ctx.job.room.sid, "job_id": ctx.job.id}


def flush_turn_metrics(chat_ctx: ChatContext) -> None:
    """Record the per-turn latency histograms of the history; called at session end."""
    job = _job_attrs()
    for msg in chat_ctx.messages():
        for name, _, value, attrs in _core.otel_turn_records(dict(msg.metrics)):
            _histogram(name).record(value, attributes={**job, **attrs})


@functools.cache
def _counter(name: str) -> Any:
    # the durations count seconds; the token and request counters have no unit
    return _meter.create_counter(name, unit="s" if name.endswith("_duration") else "")


def collect_usage(metrics: Any) -> None:
    """Record the usage counters (and connection timing) one metrics event carries."""
    job = _job_attrs()
    for name, kind, value, attrs in _core.otel_usage_records(metrics):
        if kind == "counter":
            _counter(name).add(value, attributes={**job, **attrs})
        else:
            _histogram(name).record(value, attributes={**job, **attrs})


def record_event_loop_blocked(duration: float, *, severity: str) -> None:
    attrs = _job_attrs()
    attrs["severity"] = severity
    _event_loop_blocked.record(duration, attributes=attrs)
