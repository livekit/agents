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


def _open_files() -> set[tuple[int, int]] | None:
    """Return (st_dev, st_ino) of every file this process holds open, or None
    where the platform does not list them."""
    try:
        fds = os.listdir("/dev/fd")
    except OSError:
        return None
    open_files = set()
    for fd in fds:
        try:
            st = os.fstat(int(fd))
        except (OSError, ValueError):
            continue
        open_files.add((st.st_dev, st.st_ino))
    return open_files


def _clean_multiproc_dir(path: str) -> None:
    """Remove the metric files of processes that no longer run.

    prometheus_client names each file ``<kind>_<pid>.db`` and keeps writing to it
    after it is deleted, so the collector would lose every metric of that kind.
    The files of running processes stay.

    In multiprocess mode, prometheus_client creates a file on any thread at any
    time, in the directory that PROMETHEUS_MULTIPROC_DIR names at that moment, and
    keeps it open. If that is *path*, this process's files stay, and a stale file
    from an earlier process with the same pid stays with them; clear the directory
    before the process starts to drop it. In another directory, a file with this
    process's pid is deleted only when it was already initialized and this process
    does not hold it open. Where the platform does not list open files, it stays.
    Call this before pointing PROMETHEUS_MULTIPROC_DIR at a new *path*.
    """
    own_pid = os.getpid()
    multiprocess_mode = values.ValueClass is not values.MutexValue
    current_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR") or os.environ.get(
        "prometheus_multiproc_dir"
    )
    keep_own = (
        multiprocess_mode
        and current_dir is not None
        and os.path.realpath(current_dir) == os.path.realpath(path)
    )
    to_remove: list[str] = []
    own_files: dict[str, tuple[int, int]] = {}
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        pid_str = filename.removesuffix(".db").rpartition("_")[2]
        if pid_str.isdigit():
            pid = int(pid_str)
            if pid == own_pid:
                if keep_own:
                    continue
                if multiprocess_mode:
                    try:
                        st = os.stat(file_path)
                    except OSError:
                        continue
                    if st.st_size > 0:
                        own_files[file_path] = (st.st_dev, st.st_ino)
                    continue
            elif psutil.pid_exists(pid):
                # A live process that reused a dead process's pid keeps its stale
                # file. prometheus_client's mark_process_dead has the same limit.
                continue
        to_remove.append(file_path)

    if own_files:
        # listed after every stat: a file sized before its stat was open by then
        open_files = _open_files()
        if open_files is not None:
            to_remove.extend(p for p, file_id in own_files.items() if file_id not in open_files)

    for file_path in to_remove:
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
