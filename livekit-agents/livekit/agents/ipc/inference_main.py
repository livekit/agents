from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable

from livekit import rtc

from .. import utils
from ..job import JobContext, JobProcess
from ..log import logger
from ..utils.aio import duplex_unix
from . import channel, proto


async def _async_main(
    mp_cch: socket.socket,
) -> None:
    cch = await duplex_unix._AsyncDuplex.open(mp_cch)

    exit_flag = asyncio.Event()
    no_msg_timeout = utils.aio.sleep(proto.PING_INTERVAL * 5)  # missing 5 pings

    @utils.log_exceptions(logger=logger)
    async def _read_ipc_task():
        while True:
            try:
                msg = await channel.arecv_message(cch, proto.IPC_MESSAGES)
            except duplex_unix.DuplexClosed:
                break

            with contextlib.suppress(utils.aio.SleepFinished):
                no_msg_timeout.reset()

            if isinstance(msg, proto.PingRequest):
                pong = proto.PongResponse(
                    last_timestamp=msg.timestamp, timestamp=utils.time_ms()
                )
                await channel.asend_message(cch, pong)

            if isinstance(msg, proto.ShutdownRequest):
                pass

    async def _self_health_check():
        await no_msg_timeout
        print("worker process is not responding.. worker crashed?")
        with contextlib.suppress(asyncio.CancelledError):
            exit_flag.set()

    read_task = asyncio.create_task(_read_ipc_task(), name="ipc_read")
    health_check_task = asyncio.create_task(_self_health_check(), name="health_check")

    def _done_cb(_: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            exit_flag.set()

    read_task.add_done_callback(_done_cb)

    await exit_flag.wait()
    await utils.aio.gracefully_cancel(read_task, health_check_task)

    with contextlib.suppress(duplex_unix.DuplexClosed):
        await cch.aclose()


@dataclass
class ProcStartArgs:
    log_cch: socket.socket
    mp_cch: socket.socket
    asyncio_debug: bool
