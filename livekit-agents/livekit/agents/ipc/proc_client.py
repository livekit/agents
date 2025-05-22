from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
from collections.abc import Coroutine
from typing import Callable

from ..log import logger
from ..utils import aio, log_exceptions, time_ms
from .channel import Message, arecv_message, asend_message, recv_message, send_message
from .log_queue import LogQueueHandler
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
        self._log_cch = log_cch
        self._initialize_fnc = initialize_fnc
        self._main_task_fnc = main_task_fnc
        self._initialized = False
        self._log_handler: LogQueueHandler | None = None

    def initialize_logger(self) -> None:
        if self._log_cch is None:
            raise RuntimeError("cannot initialize logger without log channel")

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.NOTSET)

        log_cch = aio.duplex_unix._Duplex.open(self._log_cch)
        self._log_handler = LogQueueHandler(log_cch)
        root_logger.addHandler(self._log_handler)

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
        aio.debug.hook_slow_callbacks(2.0)

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
            if self._log_handler is not None:
                self._log_handler.close()

            loop.run_until_complete(loop.shutdown_default_executor())

    async def send(self, msg: Message) -> None:
        await asend_message(self._acch, msg)

    async def _monitor_task(self) -> None:
        self._acch = await aio.duplex_unix._AsyncDuplex.open(self._mp_cch)
        try:
            exit_flag = asyncio.Event()
            ping_timeout = aio.sleep(self._init_req.ping_timeout)

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
