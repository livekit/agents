from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

import prometheus_client
import psutil
from opentelemetry import metrics as metrics_api
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, multiprocess

from .. import _core, utils
from ._export import worker

if TYPE_CHECKING:
    from ..llm.chat_context import ChatContext
    from ..metrics import AgentMetrics

PROC_INITIALIZE_TIME = prometheus_client.Histogram(
    "lk_agents_proc_initialize_duration_seconds",
    "Time taken to initialize a process",
    ["nodename"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)


# Use 'livesum' mode to aggregate active jobs across all processes
# This sums the values from processes that are still running
RUNNING_JOB_GAUGE = prometheus_client.Gauge(
    "lk_agents_active_job_count",
    "Active jobs",
    ["nodename"],
    multiprocess_mode="livesum",
)


# Use 'max' mode for child process count since we want the total across all processes
CHILD_PROC_GAUGE = prometheus_client.Gauge(
    "lk_agents_child_process_count",
    "Total number of child processes",
    ["nodename"],
    multiprocess_mode="max",
)


CPU_LOAD_GAUGE = prometheus_client.Gauge(
    "lk_agents_worker_load",
    "Worker load percentage",
    ["nodename"],
)


# Note: set_function() is not supported in multiprocess mode.# We need to update this metric explicitly.
def _update_child_proc_count() -> None:
    """Update child process count metric. Must be called periodically in the main process."""
    try:
        count = len(psutil.Process(os.getpid()).children(recursive=True))
        CHILD_PROC_GAUGE.labels(nodename=utils.nodename()).set(count)
    except Exception:
        # Process might not exist anymore or access denied
        pass


def _update_worker_load(worker_load: float) -> None:
    CPU_LOAD_GAUGE.labels(nodename=utils.nodename()).set(worker_load)


def job_started() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).inc()


def job_ended() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).dec()


def proc_initialized(*, time_elapsed: float) -> None:
    PROC_INITIALIZE_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)


def exposition() -> tuple[str, str]:
    """The Prometheus exposition the core's metrics listener serves, ``(content type, body)``:
    every process's metrics in multiprocess mode (``PROMETHEUS_MULTIPROC_DIR``), else this one's."""
    registry = prometheus_client.REGISTRY
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = prometheus_client.CollectorRegistry(auto_describe=True)
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    return CONTENT_TYPE_LATEST, generate_latest(registry).decode()


_meter_provider = metrics_api.get_meter_provider()
_meter = _meter_provider.get_meter("livekit-agents")


@functools.cache
def _histogram(name: str) -> metrics_api.Histogram:
    return _meter.create_histogram(name, unit="s")


# Labels are only what the core's records declare (model, agent and tool names, connection
# reuse): bounded by the application's configuration. Per-job identity stays on traces.


def flush_turn_metrics(chat_ctx: ChatContext) -> None:
    """Record the per-turn latency histograms of the history; called at session end."""
    for msg in chat_ctx.messages():
        for name, _, value, attrs in _core.otel_turn_records(dict(msg.metrics)):
            _record(name, "histogram", value, attrs)


@functools.cache
def _counter(name: str) -> metrics_api.Counter:
    # the durations count seconds; the token and request counters have no unit
    return _meter.create_counter(name, unit="s" if name.endswith("_duration") else "")


def collect_usage(metrics: AgentMetrics) -> None:
    """Record the usage counters (and connection timing) one metrics event carries."""
    for name, kind, value, attrs in _core.otel_usage_records(metrics):
        _record(name, kind, value, attrs)


def record_span_metric(name: str, value: float, attributes: dict[str, str]) -> None:
    """A duration the core measured with one of its spans (``gen_ai.invoke_agent.duration``,
    ``gen_ai.execute_tool.duration``)."""
    _record(name, "histogram", value, attributes)


def record_event_loop_blocked(duration: float, *, severity: str) -> None:
    _record("lk.agents.event_loop.blocked_duration", "histogram", duration, {"severity": severity})


def _adopt_meter_provider(provider: metrics_api.MeterProvider) -> None:
    """Resolve instruments at setup against the application's current provider."""
    global _meter, _meter_provider
    if provider is _meter_provider:
        return
    _meter_provider = provider
    _meter = provider.get_meter("livekit-agents")
    _histogram.cache_clear()
    _counter.cache_clear()


def _record(name: str, kind: str, value: float, attributes: dict[str, str]) -> None:
    if not worker().admit_metric(name, list(attributes.items())):
        return
    if kind == "counter":
        _counter(name).add(value, attributes=attributes)
    else:
        _histogram(name).record(value, attributes=attributes)
