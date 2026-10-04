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
    """Remove the metric files of processes that no longer run.

    prometheus_client names each file ``<kind>_<pid>.db`` and keeps the files of
    the current process open while it writes them. Deleting an open file would
    hide every metric of that kind from the collector, so the files of running
    processes stay. A file named after this process that it does not hold open
    is left over from an earlier process with the same pid, so it goes.
    """
    own_pid = os.getpid()
    held_open = {os.path.realpath(f.path) for f in psutil.Process(own_pid).open_files()}
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        pid_str = filename.removesuffix(".db").rpartition("_")[2]
        if pid_str.isdigit():
            pid = int(pid_str)
            if pid == own_pid:
                if os.path.realpath(file_path) in held_open:
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
