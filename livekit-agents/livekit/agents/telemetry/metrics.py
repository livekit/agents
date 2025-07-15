import prometheus_client

from .. import utils

PROC_INITIALIZE_TIME = prometheus_client.Histogram(
    "lk_agents_proc_initialize_seconds",
    "Time taken to initialize a process",
    ["nodename"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

RUNNING_JOB_GAUGE = prometheus_client.Gauge("lk_agents_running_job", "Running jobs", ["nodename"])

CHILD_PROC_GAUGE = prometheus_client.Gauge(
    "lk_agents_child_processes", "Total number of child processes", ["nodename"]
)

import psutil, os
CHILD_PROC_GAUGE.labels(nodename=utils.nodename()).set_function(
    lambda: len(psutil.Process(os.getpid()).children(recursive=True))
)


def job_started():
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).inc()


def job_ended():
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).dec()


def proc_initialized(*, time_elapsed: float):
    PROC_INITIALIZE_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)
