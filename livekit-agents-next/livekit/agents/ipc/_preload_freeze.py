"""The forkserver's last preload: it forks every job process from here, which is only safe while
it runs a single thread (a thread holding a lock at the fork leaves the lock held forever in
the child). It logs an error naming the threads a preload left running, then preserves
copy-on-write sharing for the preloaded objects: cyclic GC in a job process writes inherited
object metadata and makes those pages private.
"""

import gc
import os
import threading

from ..log import logger


def _native_threads() -> list[str]:
    """The process's threads by name, the native ones included (Linux); else by count."""
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        try:
            import psutil

            return ["thread"] * psutil.Process().num_threads()
        except Exception:
            return []
    names = []
    for tid in tasks:
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                names.append(f.read().strip())
        except OSError:
            names.append(tid)
    return names


_threads = _native_threads()
if threading.active_count() > 1 or len(_threads) > 1:
    logger.error(
        "the forkserver runs more than one thread after its preloads, the job processes it "
        "forks can deadlock: Python threads %s, OS threads %s",
        [t.name for t in threading.enumerate()],
        _threads,
    )

# Do not freeze unreachable cycles for the forkserver lifetime.
gc.collect()
gc.freeze()
