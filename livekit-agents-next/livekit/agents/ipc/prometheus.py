"""The worker's Prometheus process metrics, served on its ``prometheus`` listener."""

from __future__ import annotations

import os

import prometheus_client
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, multiprocess

from ..utils import nodename

_PROC_INITIALIZE_TIME = prometheus_client.Histogram(
    "lk_agents_proc_initialize_duration_seconds",
    "Time taken to initialize a process",
    ["nodename"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

# summed over the processes still running
_RUNNING_JOBS = prometheus_client.Gauge(
    "lk_agents_active_job_count", "Active jobs", ["nodename"], multiprocess_mode="livesum"
)

_CHILD_PROCS = prometheus_client.Gauge(
    "lk_agents_child_process_count",
    "Total number of child processes",
    ["nodename"],
    multiprocess_mode="max",
)

_WORKER_LOAD = prometheus_client.Gauge(
    "lk_agents_worker_load", "Worker load percentage", ["nodename"]
)


def update_child_proc_count() -> None:
    """Multiprocess mode has no ``set_function()``: the main process sets it periodically."""
    try:
        import psutil

        count = len(psutil.Process(os.getpid()).children(recursive=True))
        _CHILD_PROCS.labels(nodename=nodename()).set(count)
    except Exception:
        pass


def update_worker_load(worker_load: float) -> None:
    _WORKER_LOAD.labels(nodename=nodename()).set(worker_load)


def job_started() -> None:
    _RUNNING_JOBS.labels(nodename=nodename()).inc()


def job_ended() -> None:
    _RUNNING_JOBS.labels(nodename=nodename()).dec()


def proc_initialized(*, time_elapsed: float) -> None:
    _PROC_INITIALIZE_TIME.labels(nodename=nodename()).observe(time_elapsed)


def exposition() -> tuple[str, str]:
    """``(content type, body)``: every process's metrics in multiprocess mode
    (``PROMETHEUS_MULTIPROC_DIR``), else this one's."""
    registry = prometheus_client.REGISTRY
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = prometheus_client.CollectorRegistry(auto_describe=True)
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    return CONTENT_TYPE_LATEST, generate_latest(registry).decode()
