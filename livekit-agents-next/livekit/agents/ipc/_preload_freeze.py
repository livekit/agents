"""Refuse unsafe forkserver images before freezing their immutable preload data."""

import gc
import os
import threading


def _native_threads() -> list[str]:
    """The process's threads by name, the native ones included (Linux); else by count."""
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        try:
            import psutil

            return ["thread"] * psutil.Process().num_threads()
        except Exception as error:
            raise RuntimeError("cannot inspect forkserver threads; refusing to fork") from error
    names = []
    for tid in tasks:
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                names.append(f.read().strip())
        except OSError:
            names.append(tid)
    return names


# Finalizers must run before the last inspection: they can themselves start threads.
gc.collect()
_threads = _native_threads()
if not _threads:
    raise RuntimeError("cannot inspect forkserver threads; refusing to fork")
if threading.active_count() > 1 or len(_threads) > 1:
    raise RuntimeError(
        "forkserver preloads left background threads; refusing to fork: "
        f"Python threads {[t.name for t in threading.enumerate()]}, OS threads {_threads}"
    )

gc.freeze()
