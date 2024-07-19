from __future__ import annotations

import asyncio
import multiprocessing as mp

from . import channel, proto
from ..job import JobProcess
from ..log import logger
from ..utils import time_ms


async def _aproc_main(args: proto.ProcStartArgs, proc: JobProcess, cch: channel.ProcChannel) -> None:

    while True:
        try:
            msg = await cch.arecv()

            if isinstance(msg, proto.PingRequest):
                await cch.asend(
                    proto.PongResponse(last_timestamp=msg.timestamp, timestamp=time_ms())
                )


        except channel.ChannelClosed:
            break


def _proc_main(args: proto.ProcStartArgs) -> None:
    proc = mp.current_process()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    cch = channel.ProcChannel(conn=args.mp_cch, loop=loop, messages=proto.IPC_MESSAGES)
    init_req = cch.recv()
    assert isinstance(
        init_req, proto.InitializeRequest
    ), "first message must be InitializeRequest"

    logger.debug("initializing process", extra={"pid": proc.pid})
    job_proc = JobProcess()
    args.initialize_process_fnc(job_proc)
    logger.debug("process initialized", extra={"pid": proc.pid})

    # signal to the ProcPool that is worker is now ready to receive jobs
    cch.send(proto.InitializeResponse())

