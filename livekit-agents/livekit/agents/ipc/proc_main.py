from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import multiprocessing as mp
from dataclasses import dataclass

from livekit import rtc

from .. import utils
from ..job import JobContext, JobProcess
from ..log import logger
from . import channel, proto


class LogQueueHandler(logging.Handler):
    def __init__(self, queue: mp.Queue) -> None:
        super().__init__()
        self._q = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            record = copy.copy(record)
            record.message = msg
            record.msg = msg
            record.args = None
            record.exc_info = None
            record.exc_text = None
            self._q.put_nowait(record)
        except Exception:
            self.handleError(record)


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
    cch: channel.AsyncProcChannel,
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

    info = start_req.running_job
    job_ctx = JobContext(
        proc=proc,
        info=info,
        room=room,
        on_connect=_on_ctx_connect,
        on_shutdown=_on_ctx_shutdown,
    )

    @utils.log_exceptions(logger=logger)
    async def _run_job_task() -> None:
        utils.http_context._new_session_ctx()
        job_entry_task = asyncio.create_task(
            args.job_entrypoint_fnc(job_ctx), name="job_entrypoint"
        )

        async def _warn_not_connected_task():
            await asyncio.sleep(10)
            if not ctx_connect and not ctx_shutdown:
                logger.warn(
                    (
                        "room not connected after job_entry was called after 10 seconds, "
                        "did you forget to call job_ctx.connect()?"
                    )
                )

        warn_unconnected_task = asyncio.create_task(_warn_not_connected_task())
        job_entry_task.add_done_callback(lambda _: warn_unconnected_task.cancel())

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
            shutdown_tasks = []
            for callback in job_ctx._shutdown_callbacks:
                shutdown_tasks.append(
                    asyncio.create_task(callback(), name="job_shutdown_callback")
                )

            await asyncio.gather(*shutdown_tasks)
        except Exception:
            logger.exception("error while shutting down the job")

        await utils.http_context._close_http_ctx()
        exit_proc_fut.set()

    task = asyncio.create_task(_run_job_task())
    job_task = JobTask(job_ctx=job_ctx, task=task, shutdown_fut=request_shutdown_fut)
    return job_task


async def _async_main(
    args: proto.ProcStartArgs, proc: JobProcess, cch: channel.AsyncProcChannel
) -> None:
    job_task: JobTask | None = None
    exit_proc_fut = asyncio.Event()

    @utils.log_exceptions(logger=logger)
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
                        with contextlib.suppress(asyncio.InvalidStateError):
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
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    handler = LogQueueHandler(args.log_q)
    root_logger.addHandler(handler)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(args.asyncio_debug)

    cch = channel.AsyncProcChannel(
        conn=args.mp_cch, loop=loop, messages=proto.IPC_MESSAGES
    )
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
    try:
        main_task = loop.create_task(
            _async_main(args, job_proc, cch), name="job_proc_main"
        )
        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            # ignore the keyboard interrupt, we handle the process shutdown ourselves
            # (this signal can be sent by watchfiles on dev mode)
            loop.run_until_complete(main_task)
    finally:
        # try:
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.run_until_complete(cch.aclose())
        # finally:
        # loop.close()
        # pass
