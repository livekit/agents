from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
from collections.abc import Coroutine
from types import FrameType
from typing import Callable

from ..log import logger
from ..utils import aio, log_exceptions, time_ms
from .channel import Message, arecv_message, asend_message, recv_message, send_message
from .proto import (
    IPC_MESSAGES,
    InitializeRequest,
    InitializeResponse,
    PingRequest,
    PongResponse,
)


class _ProcClient:
    def __init__(
        self,
        mp_cch: socket.socket,
        log_cch: socket.socket | None,
        initialize_fnc: Callable[[InitializeRequest, _ProcClient], None],
        main_task_fnc: Callable[[aio.ChanReceiver[Message]], Coroutine[None, None, None]],
    ) -> None:
        self._mp_cch = mp_cch
        self._initialize_fnc = initialize_fnc
        self._main_task_fnc = main_task_fnc
        self._initialized = False

    def initialize(self) -> None:
        try:
            cch = aio.duplex_unix._Duplex.open(self._mp_cch)
            first_req = recv_message(cch, IPC_MESSAGES)

            assert isinstance(first_req, InitializeRequest), (
                "first message must be proto.InitializeRequest"
            )

            self._init_req = first_req
            try:
                self._initialize_fnc(self._init_req, self)
                send_message(cch, InitializeResponse())
            except Exception as e:
                send_message(cch, InitializeResponse(error=str(e)))
                raise

            self._initialized = True
            cch.detach()
        except aio.duplex_unix.DuplexClosed as e:
            raise RuntimeError("failed to initialize proc_client") from e

    def run(self) -> None:
        if not self._initialized:
            raise RuntimeError("proc_client not initialized")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_debug(self._init_req.asyncio_debug)
        loop.slow_callback_duration = 0.1  # 100ms

        try:
            self._task = loop.create_task(self._monitor_task(), name="proc_client_main")
            while not self._task.done():
                try:
                    loop.run_until_complete(self._task)
                except KeyboardInterrupt:
                    # ignore the keyboard interrupt, we handle the process shutdown ourselves on the worker process  # noqa: E501
                    # (See proto.ShutdownRequest)
                    pass

        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(loop.shutdown_default_executor())

    async def send(self, msg: Message) -> None:
        await asend_message(self._acch, msg)

    async def _monitor_task(self) -> None:
        self._acch = await aio.duplex_unix._AsyncDuplex.open(self._mp_cch)
        try:
            exit_flag = asyncio.Event()
            ping_timeout = aio.sleep(self._init_req.ping_timeout + 10)

            ipc_ch = aio.Chan[Message]()

            @log_exceptions(logger=logger)
            async def _read_ipc_task() -> None:
                while True:
                    try:
                        msg = await arecv_message(self._acch, IPC_MESSAGES)
                    except aio.duplex_unix.DuplexClosed:
                        break

                    with contextlib.suppress(aio.SleepFinished):
                        ping_timeout.reset()

                    if isinstance(msg, PingRequest):
                        await asend_message(
                            self._acch,
                            PongResponse(last_timestamp=msg.timestamp, timestamp=time_ms()),
                        )

                    ipc_ch.send_nowait(msg)

            @log_exceptions(logger=logger)
            async def _self_health_check() -> None:
                await ping_timeout
                print(
                    "worker process is not responding.. worker crashed?",
                    file=sys.stderr,
                )

            read_task = asyncio.create_task(_read_ipc_task(), name="ipc_read")
            health_check_task: asyncio.Task[None] | None = None
            if self._init_req.ping_interval > 0:
                health_check_task = asyncio.create_task(_self_health_check(), name="health_check")
            main_task = asyncio.create_task(
                self._main_task_fnc(ipc_ch), name="main_task_entrypoint"
            )

            def _done_cb(_: asyncio.Task[None]) -> None:
                with contextlib.suppress(asyncio.InvalidStateError):
                    exit_flag.set()

                ipc_ch.close()

            read_task.add_done_callback(_done_cb)
            if health_check_task is not None:
                health_check_task.add_done_callback(_done_cb)

            main_task.add_done_callback(_done_cb)

            await exit_flag.wait()
            await aio.cancel_and_wait(read_task, main_task)
            if health_check_task is not None:
                await aio.cancel_and_wait(health_check_task)

        finally:
            await self._acch.aclose()


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
