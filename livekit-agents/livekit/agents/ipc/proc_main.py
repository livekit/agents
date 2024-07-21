from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from livekit import rtc

from .. import utils
from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from . import channel, proto


@dataclass
class _ShutdownInfo:
    user_initiated: bool
    reason: str


@dataclass
class JobTask:
    job_ctx: JobContext
    task: asyncio.Task
    shutdown_fut: asyncio.Future[_ShutdownInfo]


def _start_job(
    args: proto.ProcStartArgs,
    proc: JobProcess,
    start_req: proto.StartJobRequest,
    exit_proc_fut: asyncio.Event,
    cch: channel.ProcChannel,
) -> JobTask:
    # used to warn users if none of connect/shutdown is called inside the job_entry
    ctx_connect, ctx_shutdown = False, False
    room = rtc.Room()
    request_shutdown_fut = asyncio.Future[_ShutdownInfo]()

    @room.on("disconnected")
    def _on_room_disconnected():
        with contextlib.suppress(asyncio.InvalidStateError):
            request_shutdown_fut.set_result(
                _ShutdownInfo(user_initiated=False, reason="room disconnected")
            )

    def _on_ctx_connect() -> None:
        nonlocal ctx_connect
        ctx_connect = True

    def _on_ctx_shutdown(reason: str) -> None:
        nonlocal ctx_shutdown
        ctx_shutdown = True

        with contextlib.suppress(asyncio.InvalidStateError):
            request_shutdown_fut.set_result(
                _ShutdownInfo(user_initiated=True, reason=reason)
            )

    info = RunningJobInfo(
        job=start_req.job,
        accept_args=start_req.accept_args,
        url=start_req.url,
        token=start_req.token,
    )

    job_ctx = JobContext(
        proc=proc,
        info=info,
        room=room,
        on_connect=_on_ctx_connect,
        on_shutdown=_on_ctx_shutdown,
    )

    @utils.log_exceptions(logger=logger)
    async def _run_job_task() -> None:
        job_entry_task = asyncio.create_task(
            args.job_entrypoint_fnc(job_ctx), name="job_entrypoint"
        )

        def log_exception(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception():
                logger.error(
                    "unhandled exception while running the job task",
                    exc_info=t.exception(),
                )
            elif not ctx_connect and not ctx_shutdown:
                logger.warn("job task completed without connecting or shutting down")

        job_entry_task.add_done_callback(log_exception)

        shutdown_info = await request_shutdown_fut
        await cch.asend(proto.Exiting(reason=shutdown_info.reason))
        logger.debug(
            "shutting down job task",
            extra={
                "reason": shutdown_info.reason,
                "user_initiated": shutdown_info.user_initiated,
            },
        )
        await room.disconnect()

        try:
            job_shutdown_task = asyncio.create_task(
                args.job_shutdown_fnc(job_ctx), name="job_shutdown"
            )
            await job_shutdown_task
        except Exception:
            logger.exception("error while disconnecting room")

        exit_proc_fut.set()

    task = asyncio.create_task(_run_job_task())
    job_task = JobTask(job_ctx=job_ctx, task=task, shutdown_fut=request_shutdown_fut)
    return job_task


async def _async_main(
    args: proto.ProcStartArgs, proc: JobProcess, cch: channel.ProcChannel
) -> None:
    job_task: JobTask | None = None
    exit_proc_fut = asyncio.Event()

    async def _read_ipc_task():
        nonlocal job_task
        while True:
            try:
                msg = await cch.arecv()
                if isinstance(msg, proto.PingRequest):
                    pong = proto.PongResponse(
                        last_timestamp=msg.timestamp, timestamp=utils.time_ms()
                    )
                    await cch.asend(pong)

                if isinstance(msg, proto.StartJobRequest):
                    assert job_task is None, "job task already running"
                    job_task = _start_job(args, proc, msg, exit_proc_fut, cch)

                if isinstance(msg, proto.ShutdownRequest):
                    if job_task is not None:
                        job_task.shutdown_fut.set_result(
                            _ShutdownInfo(reason=msg.reason, user_initiated=False)
                        )
                    else:
                        exit_proc_fut.set()  # there is no running job, we can exit immediately

            except channel.ChannelClosed:
                logger.exception("channel closed, exiting")
                break

    read_task = asyncio.create_task(_read_ipc_task())
    await exit_proc_fut.wait()
    await utils.aio.gracefully_cancel(read_task)


def main(args: proto.ProcStartArgs) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    cch = channel.ProcChannel(conn=args.mp_cch, loop=loop, messages=proto.IPC_MESSAGES)
    init_req = loop.run_until_complete(cch.arecv())

    assert isinstance(
        init_req, proto.InitializeRequest
    ), "first message must be InitializeRequest"

    job_proc = JobProcess(start_arguments=args.user_arguments)
    logger.debug("initializing process", extra={"pid": job_proc.pid})
    args.initialize_process_fnc(job_proc)
    logger.debug("process initialized", extra={"pid": job_proc.pid})

    # signal to the ProcPool that is worker is now ready to receive jobs
    loop.run_until_complete(cch.asend(proto.InitializeResponse()))
    loop.run_until_complete(_async_main(args, job_proc, cch))
    loop.run_until_complete(cch.aclose())
