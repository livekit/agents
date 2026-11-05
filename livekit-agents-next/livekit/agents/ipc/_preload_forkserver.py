"""Import the forkserver image in order, refusing failures before any job can fork."""

from __future__ import annotations

import gc
import os
import threading

from . import _FORKSERVER_PRELOAD_MODULES, load_preload_modules

load_preload_modules(
    [
        "livekit.agents.ipc._preload",
        *filter(None, os.environ[_FORKSERVER_PRELOAD_MODULES].split(os.pathsep)),
    ]
)


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


# The image is frozen only after every framework, plugin and application import above.
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
