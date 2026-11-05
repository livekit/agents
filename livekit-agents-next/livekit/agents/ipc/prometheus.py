"""The worker's Prometheus process metrics, served on its ``prometheus`` listener."""

from __future__ import annotations

import os

import prometheus_client
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, multiprocess

from .. import _core
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


def process_event(event: str, process: _core.JobSupervisor) -> None:
    """The pool's notification of a job process's step (a job thread is not counted): its
    initialization time once ready, and the jobs running, from their launch to the process's
    cleanup."""
    if process.thread:
        return
    labels = {"nodename": nodename()}
    if event == "process_ready" and (elapsed := process.initialize_time) is not None:
        _PROC_INITIALIZE_TIME.labels(**labels).observe(elapsed)
    elif event == "process_job_launched":
        _RUNNING_JOBS.labels(**labels).inc()
    elif event == "process_closed" and process.running_job is not None:
        _RUNNING_JOBS.labels(**labels).dec()


def exposition() -> tuple[str, str]:
    """``(content type, body)``: every process's metrics in multiprocess mode
    (``PROMETHEUS_MULTIPROC_DIR``), else this one's."""
    registry = prometheus_client.REGISTRY
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = prometheus_client.CollectorRegistry(auto_describe=True)
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    return CONTENT_TYPE_LATEST, generate_latest(registry).decode()
