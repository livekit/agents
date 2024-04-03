from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from livekit import rtc

from .. import aio, apipe, ipc_enc
from ..job_context import JobContext
from ..utils import time_ms
from . import protocol


class LogHandler(logging.Handler):
    """Log handler forwarding logs to the worker process"""

    def __init__(self, writer: ipc_enc.ProcessPipeWriter) -> None:
        super().__init__(logging.NOTSET)
        self._writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ipc_enc.write_msg(
                self._writer,
                protocol.Log(level=record.levelno, message=record.getMessage()),
            )
        except Exception as e:
            print(f"failed to write log: {e}")


async def _start(
    pipe: apipe.AsyncPipe, args: protocol.JobMainArgs, room: rtc.Room
) -> None:
    close_tx, close_rx = aio.channel()  # used by the JobContext to signal shutdown

    cnt = room.connect(args.url, args.token)
    start_req: protocol.StartJobRequest | None = None
    usertask: asyncio.Task | None = None
    shutting_down = False

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
                shutting_down = True
                break
            if isinstance(msg, protocol.StartJobRequest):
                start_req = msg
                await _start_if_valid()
            if isinstance(msg, protocol.Ping):
                last_timestamp = msg.timestamp
                await pipe.write(
                    protocol.Pong(last_timestamp=last_timestamp, timestamp=time_ms())
                )

    logging.debug("disconnecting from room")
    await room.disconnect()

    if usertask is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await usertask  # type: ignore

    if shutting_down:
        await pipe.write(protocol.ShutdownResponse())


def _run_job(cch: ipc_enc.ProcessPipe, args: protocol.JobMainArgs) -> None:
    """Entry point for a job process"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logging.root.setLevel(logging.NOTSET)
    # logging.root.propagate = False
    logging.root.addHandler(LogHandler(cch))

    # current process pid
    pid = os.getpid()
    logging.debug(
        "process started", extra={"job_id": args.job_id, "url": args.url, "pid": pid}
    )

    pipe = apipe.AsyncPipe(cch, loop, protocol.IPC_MESSAGES)
    loop.slow_callback_duration = 0.02  # 20ms
    aio.debug.hook_slow_callbacks(0.75)
    loop.set_debug(args.asyncio_debug)

    room = rtc.Room(loop=loop)
    main_task = loop.create_task(_start(pipe, args, room))

    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        # ignore
        loop.run_until_complete(main_task)
