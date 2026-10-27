from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from types import FrameType
from typing import Any

from .. import _core
from ..log import logger
from ..telemetry import loop_monitor
from ..utils import log_exceptions


def log_to_stdout(
    config: tuple[str, str] | None, levels: dict[str, int], fields: dict[str, Any]
) -> None:
    """A child process writes its own log lines, as the worker does (``config``: its level and
    line style, none when the worker set up no logging), to the stdout it inherits; each record
    stamped with ``fields``, the process's (its pid, the job's ids once one runs)."""
    from ..cli import setup_logging

    if config is not None:
        setup_logging(*config).addFilter(_Stamp(fields))
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


class _Stamp(logging.Filter):
    def __init__(self, fields: dict[str, Any]) -> None:
        super().__init__()
        self._fields = fields

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__.update(self._fields)
        return True


def serve_parent(mp_cch: socket.socket, host: Any, *, job: bool) -> None:
    """A child process's (or job thread's) side of the parent's socket: the handshake
    (``host.initialize(http_proxy)``), then a loop of its own on which ``_core.IpcClient.serve``
    hands the parent's messages to ``host`` until the parent goes or the child is done. A failed
    initialization returns at once (the parent was told)."""
    try:
        client = _core.IpcClient(mp_cch.detach(), host.initialize, job=job)
    except Exception:
        return
    host._client = client

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(client.asyncio_debug)
    loop.slow_callback_duration = 0.1  # 100ms
    # flag synchronous code blocking the job loop (spans + warnings), see telemetry.loop_monitor
    loop_monitor.start_monitoring(loop, name="job")

    try:
        task = loop.create_task(_serve(client, host), name="proc_client_main")
        while not task.done():
            # the parent handles the process's shutdown (its ShutdownRequest)
            with contextlib.suppress(KeyboardInterrupt):
                loop.run_until_complete(task)
    finally:
        loop_monitor.stop_monitoring(loop)
        # shut the loop down the way asyncio.run() does, a thread runner reuses the process
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@log_exceptions(logger=logger)
async def _serve(client: _core.IpcClient, host: Any) -> None:
    try:
        await client.serve(host)
    finally:
        await host.aclose()
        await client.aclose()


def _dump_stack_traces_impl() -> None:
    """Implementation of stack trace dumping (callable directly or from signal handler)."""
    import asyncio
    import faulthandler
    import os
    import tempfile
    import time
    import traceback
    from multiprocessing import current_process
    from pathlib import Path

    import psutil

    if os.getenv("LK_DUMP_STACK_TRACES", "0").lower() in ("0", "false", "no"):
        return

    dir: str = os.getenv("LK_DUMP_DIR", tempfile.gettempdir())
    Path(dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=dir,
        delete=False,
        prefix=f"livekit-agents-pid-{current_process().pid}-{time.time_ns()}-",
        suffix=".stacktrace",
    ) as f:
        print(f"\n{'=' * 60}", file=f)
        print(
            f"Process {current_process().name} (pid {current_process().pid}) stack trace dump",
            file=f,
        )
        print(f"{'=' * 60}\n", file=f)

        faulthandler.dump_traceback(file=f, all_threads=True)
        print("\n", file=f)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                print("=" * 60, file=f)
                print("ASYNCIO TASKS", file=f)
                print("=" * 60, file=f)

                tasks = asyncio.all_tasks(loop)
                print(f"Total tasks: {len(tasks)}\n", file=f)

                for i, task in enumerate(tasks, 1):
                    print(f"\n--- Task {i}/{len(tasks)} ---", file=f)
                    print(f"Name: {task.get_name()}", file=f)
                    print(f"Done: {task.done()}", file=f)

                    if not task.done():
                        print(f"Cancelled: {task.cancelled()}", file=f)

                        try:
                            stack = task.get_stack()
                            print(f"Stack frames: {len(stack)}", file=f)
                            print("Stack trace:", file=f)
                            for frame in stack:
                                traceback.print_stack(frame, limit=1, file=f)
                        except Exception as e:
                            print(f"Could not get stack: {e}", file=f)

                        try:
                            coro = task.get_coro()
                            print(f"Coroutine: {coro}", file=f)
                            if cr_frame := getattr(coro, "cr_frame", None):
                                print("Coroutine frame:", file=f)
                                traceback.print_stack(cr_frame, file=f)
                        except Exception as e:
                            print(f"Could not get coroutine: {e}", file=f)
                    else:
                        try:
                            exc = task.exception()
                            if exc:
                                print(f"Exception: {exc}", file=f)
                                print("Exception traceback:", file=f)
                                traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
                        except Exception as e:
                            print(f"Could not get exception: {e}", file=f)

                    print("", file=f)
            else:
                print("No asyncio event loop running", file=f)
        except Exception as e:
            print(f"Error dumping asyncio tasks: {e}", file=f)
            traceback.print_exc(file=f)

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            print("\n" + "=" * 60, file=f)
            print("MEMORY USAGE", file=f)
            print("=" * 60, file=f)
            print(f"RSS: {memory_mb:.2f} MB", file=f)
            print(f"VMS: {memory_info.vms / (1024 * 1024):.2f} MB", file=f)
        except Exception:
            pass


def _dump_stack_traces(signum: int, _: FrameType | None) -> None:
    """Signal handler wrapper for _dump_stack_traces_impl."""
    _dump_stack_traces_impl()
