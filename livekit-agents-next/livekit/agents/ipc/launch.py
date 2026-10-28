"""Multiprocessing launch integration, including ownership of partially constructed Popen objects."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from ..job import JobProcess

_FORKSERVER_PRELOAD_MODULES = "LIVEKIT_AGENTS_FORKSERVER_PRELOAD_MODULES"


class _ForkServerState(Protocol):
    # CPython publishes these launch-integration fields, but typeshed omits them.
    _forkserver_pid: int | None
    _preload_modules: list[str]


def validate_setup_fnc(
    setup_fnc: Callable[[JobProcess], None] | None,
) -> Callable[[JobProcess], None] | None:
    """Process setup is synchronous because initialization precedes the job event loop."""
    if setup_fnc is not None and (
        not callable(setup_fnc)
        or inspect.iscoroutinefunction(setup_fnc)
        or inspect.isasyncgenfunction(setup_fnc)
        or inspect.iscoroutinefunction(type(setup_fnc).__call__)
        or inspect.isasyncgenfunction(type(setup_fnc).__call__)
    ):
        raise TypeError("setup_fnc must be a synchronous callable")
    return setup_fnc


def preload_module_names(modules: Sequence[str]) -> tuple[str, ...]:
    """Validate the launch configuration before allocating a child process."""
    if isinstance(modules, str):
        raise TypeError("preload_modules must be a sequence of module names, not a string")
    names = tuple(modules)
    for module in names:
        if not isinstance(module, str) or not all(
            part.isidentifier() for part in module.split(".")
        ):
            raise ValueError(f"invalid preload module name {module!r}")
    return names


def load_preload_modules(modules: Sequence[str]) -> None:
    """Import in order and retain the failing module and original cause."""
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            # multiprocessing's forkserver ignores ImportError from its preload list.
            raise RuntimeError(f"preload module {module!r} failed: {error}") from error


def configure_forkserver(context: BaseContext, modules: Sequence[str]) -> None:
    """Install one fail-closed preload entrypoint before the forkserver starts."""
    modules = preload_module_names(modules)
    from multiprocessing.forkserver import _forkserver

    state = cast(_ForkServerState, _forkserver)
    # Retain the ordered configuration in multiprocessing's own preload inventory. The
    # bootstrap already imports these modules, so its trailing imports are cached no-ops.
    inventory = ["livekit.agents.ipc._preload_forkserver", *modules]
    if state._forkserver_pid is not None:
        if state._preload_modules != inventory:
            raise RuntimeError(
                "the running forkserver has different preload_modules; configure all servers "
                "with the same ordered preloads before launching jobs, or use spawn"
            )
        return
    # The forkserver is a fresh interpreter; only its environment survives that exec.
    os.environ[_FORKSERVER_PRELOAD_MODULES] = os.pathsep.join(modules)
    context.set_forkserver_preload(inventory)


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
