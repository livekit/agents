import os

import prometheus_client
import psutil
from prometheus_client import values

from .. import utils
from ..log import logger

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


def _clean_multiproc_dir(path: str) -> None:
    """Remove the metric files of processes that no longer run.

    prometheus_client names each file ``<kind>_<pid>.db`` and keeps writing to it
    after it is deleted, so the collector would lose every metric of that kind.
    The files of running processes stay. In multiprocess mode, prometheus_client
    creates each new file on any thread at any time, in the directory that
    PROMETHEUS_MULTIPROC_DIR names at that moment. If that is *path*, this
    process's own files stay too, and a stale file from an earlier process with
    the same pid stays with them; clear the directory before the process starts
    to drop it. Otherwise this process has written no file in *path*, so any file
    with its pid is stale. Call this before pointing PROMETHEUS_MULTIPROC_DIR at
    a new *path*.
    """
    own_pid = os.getpid()
    current_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR") or os.environ.get(
        "prometheus_multiproc_dir"
    )
    writes_files = (
        values.ValueClass is not values.MutexValue
        and current_dir is not None
        and os.path.realpath(current_dir) == os.path.realpath(path)
    )
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        pid_str = filename.removesuffix(".db").rpartition("_")[2]
        if pid_str.isdigit():
            pid = int(pid_str)
            if pid == own_pid:
                if writes_files:
                    continue
            elif psutil.pid_exists(pid):
                # A live process that reused a dead process's pid keeps its stale
                # file. prometheus_client's mark_process_dead has the same limit.
                continue

        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"failed to remove {file_path}", exc_info=e)


def _update_worker_load(worker_load: float) -> None:
    CPU_LOAD_GAUGE.labels(nodename=utils.nodename()).set(worker_load)


def job_started() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).inc()


def job_ended() -> None:
    RUNNING_JOB_GAUGE.labels(nodename=utils.nodename()).dec()


def proc_initialized(*, time_elapsed: float) -> None:
    PROC_INITIALIZE_TIME.labels(nodename=utils.nodename()).observe(time_elapsed)
