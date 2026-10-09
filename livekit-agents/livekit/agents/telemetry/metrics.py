import os

import prometheus_client
import psutil

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
    """Remove the metric files of processes that no longer run, and the files with
    this process's pid that it does not have open.

    prometheus_client keeps each file open and keeps writing to it after it is
    deleted, so the collector would lose every metric of that kind. The directory
    is listed before the open files are, so a file created during the cleanup is
    either not listed or already open.
    """
    own_pid = os.getpid()
    filenames = os.listdir(path)
    open_paths = {os.path.realpath(f.path) for f in psutil.Process().open_files()}
    for filename in filenames:
        file_path = os.path.join(path, filename)
        if os.path.realpath(file_path) in open_paths:
            continue
        try:
            pid = int(filename.removesuffix(".db").rpartition("_")[2])
            # prometheus_client's mark_process_dead has the same limit: a live process
            # that reused a dead process's pid keeps the dead process's file
            other_running = pid != own_pid and psutil.pid_exists(pid)
        except (ValueError, OverflowError):  # not a pid
            other_running = False
        if other_running:
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
