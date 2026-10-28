"""Multiprocessing launch integration, including ownership of partially constructed Popen objects."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.process import BaseProcess


@dataclass(frozen=True)
class ProcessReceipt:
    pid: int
    sentinel: int
    owned_handles: list[int]


def start_process(
    process: BaseProcess, method: str, register: Callable[[ProcessReceipt], None]
) -> None:
    if sys.platform == "win32":
        from multiprocessing.popen_spawn_win32 import Popen
    elif method == "forkserver":
        from multiprocessing.popen_forkserver import Popen
    else:
        # Both POSIX concrete classes have the same bootstrap contract.
        from multiprocessing.popen_spawn_posix import Popen  # type: ignore[assignment]

    class OwnedPopen(Popen):
        def __init__(self, process_obj: BaseProcess) -> None:
            try:
                super().__init__(process_obj)
            finally:
                # Upstream may fail after OS launch but before BaseProcess.start assigns Popen.
                # No asynchronous hop or descriptor allocation can intervene in this handoff.
                try:
                    sentinel = self.sentinel
                    finalizer = self.finalizer
                except AttributeError:
                    pass  # No process identity was published by this start method.
                else:
                    try:
                        pid = self.pid
                    except AttributeError:
                        if method != "forkserver":
                            raise
                        pid = 0
                    process_obj._popen = self  # type: ignore[attr-defined]
                    handles: Sequence[int] = finalizer._args  # type: ignore[attr-defined]
                    owned = [handle for handle in handles if handle != sentinel]
                    finalizer.cancel()
                    register(ProcessReceipt(pid, sentinel, owned))

    def factory(process_obj: BaseProcess) -> OwnedPopen:
        # The temporary factory contains a native handle and must never enter the child pickle.
        del process_obj._Popen  # type: ignore[attr-defined]
        return OwnedPopen(process_obj)

    process._Popen = factory  # type: ignore[attr-defined]
    try:
        process.start()
    finally:
        # Native supervision is the only waiter, including when Popen.__init__ raised.
        from multiprocessing.process import _children  # type: ignore[attr-defined]

        _children.discard(process)
