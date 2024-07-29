from __future__ import annotations
from .. import utils

import prometheus_client


PROC_PREWARM_TIME = prometheus_client.Histogram(
    "lk_agents_proc_warm_time_seconds",
    "Time taken to warm a process",
    ["nodename"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

PROC_STATUS_GAUGE = prometheus_client.Gauge(
    "lk_agents_running_job", "Running jobs", ["nodename"]
)


def job_started():
    PROC_STATUS_GAUGE.labels(nodename=utils.nodename()).inc()


def job_ended():
    PROC_STATUS_GAUGE.labels(nodename=utils.nodename()).dec()


def proc_initialized(time_elapsed: float):
    PROC_PREWARM_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)
