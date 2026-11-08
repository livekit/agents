from __future__ import annotations

import asyncio
import importlib
import logging
import os
import socket
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from types import ModuleType
from typing import Any, Protocol, cast

from .. import _core
from .._generated import public_ipc as _public
from .._generated.public_ipc import *  # noqa: F403
from .._generated.worker_ipc import JobCompletion
from ..job import JobContext
from ..log import logger
from . import child


class _JobThread:
    def __init__(self, args: child.JobArgs, launch: _core.ThreadLaunch) -> None:
        def run() -> None:
            try:
                child.thread_main(args)
            finally:
                launch.thread_exited()

        self._thread = threading.Thread(target=run, name="job_thread_runner")

    def start(self) -> None:
        self._thread.start()


class JobLauncher:
    """Starts one job runner: a process of ``mp_ctx``'s start method, or a thread."""

    def __init__(
        self,
        *,
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        session_end_timeout: float,
        preload_modules: Sequence[str] = (),
        env: Sequence[tuple[str, str]] = (),
        mp_ctx: BaseContext | None = None,
        user_arguments: Any | None = None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._mp_ctx = mp_ctx
        self._proc: Any = None
        self._args = child.JobArgs(
            job_entrypoint_fnc=job_entrypoint_fnc,
            session_end_fnc=session_end_fnc,
            simulation_end_fnc=simulation_end_fnc,
            session_end_timeout=session_end_timeout,
            preload_modules=tuple(preload_modules),
            user_arguments=user_arguments,
            env=tuple(env),
        )

    async def _core_launch(
        self, endpoint: int, launch: _core.ProcessLaunch | _core.ThreadLaunch
    ) -> None:
        child_end = child.PipeEnd(endpoint) if os.name == "nt" else socket.socket(fileno=endpoint)
        if isinstance(launch, _core.ThreadLaunch):
            try:
                self._proc = _JobThread(
                    child.JobArgs(**{**vars(self._args), "mp_cch": child_end}), launch
                )
                self._proc.start()
            except BaseException:
                child_end.close()
                raise
            return
        try:
            if self._mp_ctx is None:
                raise ValueError("a job process needs a multiprocessing context")
            self._proc = self._create_process(self._mp_ctx, child_end)
            method = self._mp_ctx.get_start_method()

            def register(pid: int, sentinel: int, owned_handles: list[int]) -> None:
                launch.process_started(pid, sentinel, method, owned_handles)

            # Off the loop so the parent keeps handling shutdown signals while children start.
            await self._loop.run_in_executor(None, start_process, self._proc, method, register)
        finally:
            # The child process holds its own copy.
            child_end.close()

    async def _core_completed(self, completion: JobCompletion) -> None:
        if isinstance(self._proc, BaseProcess):
            # Bookkeeping only: the core has already reaped the child.
            self._proc._popen.returncode = completion.exit_code  # type: ignore[attr-defined]
            self._proc._popen.finalizer()  # type: ignore[attr-defined]

    def _create_process(
        self, mp_ctx: BaseContext, cch: socket.socket | child.PipeEnd
    ) -> BaseProcess:
        from ..cli import _log_config

        levels = {"root": logging.getLogger().level}
        for log in logging.Logger.manager.loggerDict.values():
            if isinstance(log, logging.Logger):
                levels[log.name] = log.level
        self._args.mp_cch = cch
        self._args.log_config = _log_config
        self._args.logger_levels = levels
        return cast(
            BaseProcess,
            mp_ctx.Process(  # type: ignore[attr-defined]
                target=child.job_main, args=(self._args,), name="job_proc"
            ),
        )


_FORKSERVER_PRELOAD_MODULES = "LIVEKIT_AGENTS_FORKSERVER_PRELOAD_MODULES"


class _ForkServerState(Protocol):
    # CPython publishes these launch-integration fields, but typeshed omits them.
    _forkserver_pid: int | None
    _preload_modules: list[str]


def preload_module_names(modules: Sequence[str]) -> tuple[str, ...]:
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
    # Plugin lazy exports are resolved too, so a job's first use imports nothing on its loop.
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            # multiprocessing's forkserver ignores ImportError from its preload list.
            raise RuntimeError(f"preload module {module!r} failed: {error}") from error
    from ..plugin import Plugin

    for plugin in Plugin.registered_plugins:
        _load_exports(importlib.import_module(plugin.package))


def _load_exports(module: ModuleType) -> None:
    # An export whose optional dependency is missing stays unresolved; it fails only when used.
    for name in getattr(module, "__all__", ()):
        try:
            value = getattr(module, name)
        except ImportError as error:
            logger.debug("plugin export not preloaded", extra={"export": name, "error": error})
            continue
        if isinstance(value, ModuleType) and value.__name__.startswith(module.__name__ + "."):
            _load_exports(value)


def configure_forkserver(context: BaseContext, modules: Sequence[str]) -> None:
    modules = preload_module_names(modules)
    from multiprocessing.forkserver import _forkserver

    state = cast(_ForkServerState, _forkserver)
    # _preload_forkserver already imports `modules` (failing closed), so the trailing entries
    # are cached no-ops kept only for the consistency check below.
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


def start_process(
    process: BaseProcess, method: str, register: Callable[[int, int, list[int]], None]
) -> None:
    if sys.platform == "win32":
        from multiprocessing.popen_spawn_win32 import Popen
    elif method == "forkserver":
        from multiprocessing.popen_forkserver import Popen
    else:
        from multiprocessing.popen_spawn_posix import Popen  # type: ignore[assignment]

    class OwnedPopen(Popen):
        def __init__(self, process_obj: BaseProcess) -> None:
            try:
                super().__init__(process_obj)
            finally:
                # Upstream may fail after the OS launch but before BaseProcess.start assigns
                # Popen; hand any started process to the core regardless.
                try:
                    sentinel = self.sentinel
                    finalizer = self.finalizer
                except AttributeError:
                    pass  # no process was started
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
                    register(pid, sentinel, owned)

    def factory(process_obj: BaseProcess) -> OwnedPopen:
        # The factory holds a native handle and must never enter the child pickle.
        del process_obj._Popen  # type: ignore[attr-defined]
        return OwnedPopen(process_obj)

    process._Popen = factory  # type: ignore[attr-defined]
    try:
        process.start()
    finally:
        # The core is the only waiter, even when Popen.__init__ raised.
        from multiprocessing.process import _children  # type: ignore[attr-defined]

        _children.discard(process)


__all__ = [
    "JobLauncher",
]
__all__ += _public.__all__
