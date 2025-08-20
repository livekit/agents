import os

import prometheus_client
import psutil

from .. import utils

PROC_INITIALIZE_TIME = prometheus_client.Histogram(
    "lk_agents_proc_initialize_duration_seconds",
    "Time taken to initialize a process",
    ["nodename"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

RUNNING_JOB_GAUGE = prometheus_client.Gauge(
    "lk_agents_active_job_count", "Active jobs", ["nodename"]
)

CHILD_PROC_GAUGE = prometheus_client.Gauge(
    "lk_agents_child_process_count", "Total number of child processes", ["nodename"]
)


CHILD_PROC_GAUGE.labels(nodename=utils.nodename()).set_function(
    lambda: len(psutil.Process(os.getpid()).children(recursive=True))
)


def job_started() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).inc()


def job_ended() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).dec()


def proc_initialized(*, time_elapsed: float) -> None:
    PROC_INITIALIZE_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)
