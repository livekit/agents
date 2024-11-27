import asyncio
import contextlib
import socket
import sys
from typing import Callable, Coroutine

from ..log import logger
from ..utils import aio, log_exceptions, time_ms
from . import proto
from .channel import Message, arecv_message, asend_message, recv_message, send_message


class _ProcClient:
    def __init__(
        self,
        mp_cch: socket.socket,
        initialize_fnc: Callable[[proto.InitializeRequest, "_ProcClient"], None],
        entrypoint_fnc: Callable[
            [aio.ChanReceiver[Message]], Coroutine[None, None, None]
        ],
        asyncio_debug: bool,
    ) -> None:
        self._mp_cch = mp_cch
        self._asyncio_debug = asyncio_debug
        self._initialize_fnc = initialize_fnc
        self._entrypoint_fnc = entrypoint_fnc
        self._initialized = False

    def initialize(self) -> None:
        try:
            cch = aio.duplex_unix._Duplex.open(self._mp_cch)
            self._init_req = recv_message(cch, proto.IPC_MESSAGES)

            assert isinstance(
                self._init_req, proto.InitializeRequest
            ), "first message must be proto.InitializeRequest"

            self._initialize_fnc(self._init_req, self)
            send_message(cch, proto.InitializeResponse())
            self._initialized = True
            cch.detach()
        except aio.duplex_unix.DuplexClosed as e:
            raise RuntimeError("failed to initialize proc_client") from e

    def run(self) -> None:
        if not self._initialized:
            raise RuntimeError("proc_client not initialized")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_debug(self._asyncio_debug)
        loop.slow_callback_duration = 0.1  # 100ms
        aio.debug.hook_slow_callbacks(2.0)

        try:
            self._task = loop.create_task(self._main_task(), name="proc_client_main")
            while not self._task.done():
                try:
                    loop.run_until_complete(self._task)
                except KeyboardInterrupt:
                    # ignore the keyboard interrupt, we handle the process shutdown ourselves on the worker process
                    # (See proto.ShutdownRequest)
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            loop.run_until_complete(loop.shutdown_default_executor())

    async def send(self, msg: Message) -> None:
        await asend_message(self._acch, msg)

    async def _main_task(self) -> None:
        self._acch = await aio.duplex_unix._AsyncDuplex.open(self._mp_cch)
        try:
            exit_flag = asyncio.Event()
            ping_timeout = aio.sleep(proto.PING_INTERVAL * 5)

            ipc_ch = aio.Chan[Message]()

            @log_exceptions(logger=logger)
            async def _read_ipc_task():
                while True:
                    try:
                        msg = await arecv_message(self._acch, proto.IPC_MESSAGES)
                    except aio.duplex_unix.DuplexClosed:
                        break

                    with contextlib.suppress(aio.SleepFinished):
                        ping_timeout.reset()

                    if isinstance(msg, proto.PingRequest):
                        await asend_message(
                            self._acch,
                            proto.PongResponse(
                                last_timestamp=msg.timestamp, timestamp=time_ms()
                            ),
                        )

                    ipc_ch.send_nowait(msg)

            async def _self_health_check():
                await ping_timeout
                print(
                    "worker process is not responding.. worker crashed?",
                    file=sys.stderr,
                )

            read_task = asyncio.create_task(_read_ipc_task(), name="ipc_read")
            health_check_task = asyncio.create_task(
                _self_health_check(), name="health_check"
            )
            entrypoint_task = asyncio.create_task(
                self._entrypoint_fnc(ipc_ch), name="entrypoint"
            )

            def _done_cb(_: asyncio.Task) -> None:
                with contextlib.suppress(asyncio.InvalidStateError):
                    exit_flag.set()

                ipc_ch.close()

            read_task.add_done_callback(_done_cb)
            health_check_task.add_done_callback(_done_cb)
            entrypoint_task.add_done_callback(_done_cb)

            await exit_flag.wait()
            await aio.gracefully_cancel(read_task, health_check_task, entrypoint_task)
        finally:
            await self._acch.aclose()
