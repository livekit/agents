from __future__ import annotations

import asyncio
import contextlib
import logging

from livekit import rtc

from .. import aio
from ..job_context import JobContext
from ..log import logger
from ..utils import time_ms
from . import apipe, protocol


class LogHandler(logging.Handler):
    """Log handler forwarding logs to the worker process"""

    def __init__(self, writer: protocol.ProcessPipeWriter) -> None:
        super().__init__(logging.NOTSET)
        self._writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        protocol.write_msg(
            self._writer,
            protocol.Log(level=record.levelno, message=record.getMessage()),
        )


async def _start(
    pipe: apipe.AsyncPipe,
    args: protocol.JobMainArgs,
    room=rtc.Room(),
) -> None:
    close_tx, close_rx = aio.channel()  # used by the JobContext to signal shutdown

    cnt = room.connect(args.url, args.token)
    start_req: protocol.StartJobRequest | None = None
    usertask: asyncio.Task | None = None

    async def _start_if_valid():
        nonlocal usertask
        if start_req and room.isconnected():
            # start the job
            await pipe.write(protocol.StartJobResponse())

            ctx = JobContext(
                close_tx,
                start_req.job,
                room,
            )
            usertask = asyncio.create_task(args.target(ctx))

    async with contextlib.aclosing(aio.select([pipe, cnt, close_rx])) as select:
        while True:
            s = await select()
            if s.selected is cnt:
                if s.exc:
                    await pipe.write(protocol.StartJobResponse(exc=s.exc))
                    break  # failed to connect, break and exit the process
                await _start_if_valid()

            if s.selected is close_rx:
                await pipe.write(protocol.UserExit())
                break

            msg = s.result()
            if isinstance(msg, protocol.ShutdownRequest):
                await pipe.write(protocol.ShutdownResponse())
                break
            if isinstance(msg, protocol.StartJobRequest):
                start_req = msg
                await _start_if_valid()
            if isinstance(msg, protocol.Ping):
                await pipe.write(
                    protocol.Pong(last_timestamp=msg.timestamp, timestamp=time_ms())
                )

    await room.disconnect()

    if usertask is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await usertask  # type: ignore


def _run_job(cch: protocol.ProcessPipe, args: protocol.JobMainArgs) -> None:
    """Entry point for a job process"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logging.basicConfig(handlers=[LogHandler(cch)], level=logging.NOTSET)
    logger.debug("process started")

    pipe = apipe.AsyncPipe(cch, loop=loop)
    loop.slow_callback_duration = 0.01  # 10ms
    loop.run_until_complete(_start(pipe, args))
