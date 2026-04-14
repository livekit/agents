import os

from opentelemetry import metrics as metrics_api
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
import prometheus_client
import psutil

from .. import utils

_meter = metrics_api.get_meter("livekit-agent-server")

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

OTEL_PROC_INITIALIZE_TIME = _meter.create_histogram(
    "lk.agents.server.proc_initialize_time",
    unit="s",
    description="Time taken to initialize a worker process",
)
OTEL_RUNNING_JOB_COUNTER = _meter.create_up_down_counter(
    "lk.agents.server.active_job_count",
    description="Active jobs on the agent server",
)


class _AgentServerMetricsState:
    def __init__(self) -> None:
        self.child_process_count = 0
        self.worker_load = 0.0


_agent_server_metrics_state = _AgentServerMetricsState()


def setup_worker_observability_metrics() -> None:
    current_meter_provider = metrics_api.get_meter_provider()
    if isinstance(current_meter_provider, SdkMeterProvider):
        return

    metric_exporter = OTLPMetricExporter()
    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
    meter_provider = SdkMeterProvider(
        resource=Resource.create(
            {SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "livekit-agent-server")}
        ),
        metric_readers=[reader],
    )
    metrics_api.set_meter_provider(meter_provider)


def _node_attrs() -> dict[str, str]:
    return {"nodename": utils.nodename()}


def _observe_child_process_count() -> list[metrics_api.Observation]:
    return [metrics_api.Observation(_agent_server_metrics_state.child_process_count, attributes=_node_attrs())]


def _observe_worker_load() -> list[metrics_api.Observation]:
    return [metrics_api.Observation(_agent_server_metrics_state.worker_load, attributes=_node_attrs())]


_meter.create_observable_gauge(
    "lk.agents.server.child_process_count",
    callbacks=[lambda options: _observe_child_process_count()],
    description="Child process count on the agent server",
)
_meter.create_observable_gauge(
    "lk.agents.server.worker_load",
    callbacks=[lambda options: _observe_worker_load()],
    description="Worker load percentage",
)


# Note: set_function() is not supported in multiprocess mode.# We need to update this metric explicitly.
def _update_child_proc_count() -> None:
    """Update child process count metric. Must be called periodically in the main process."""
    try:
        count = len(psutil.Process(os.getpid()).children(recursive=True))
        _agent_server_metrics_state.child_process_count = count
        CHILD_PROC_GAUGE.labels(nodename=utils.nodename()).set(count)
    except Exception:
        # Process might not exist anymore or access denied
        pass


def _update_worker_load(worker_load: float) -> None:
    _agent_server_metrics_state.worker_load = worker_load
    CPU_LOAD_GAUGE.labels(nodename=utils.nodename()).set(worker_load)


def job_started() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).inc()
    OTEL_RUNNING_JOB_COUNTER.add(1, attributes=_node_attrs())


def job_ended() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).dec()
    OTEL_RUNNING_JOB_COUNTER.add(-1, attributes=_node_attrs())


def proc_initialized(*, time_elapsed: float) -> None:
    PROC_INITIALIZE_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)
    OTEL_PROC_INITIALIZE_TIME.record(time_elapsed, attributes=_node_attrs())
