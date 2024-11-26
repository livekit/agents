from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable

from livekit import rtc

from .. import utils
from ..job import JobContext, JobProcess, _JobContextVar
from ..log import logger
from ..utils.aio import duplex_unix
from . import channel, proto

from .inference_executor import InferenceExecutor, _InferenceRunnerClient


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
    proc: JobProcess,
    job_entrypoint_fnc: Callable[[JobContext], Any],
    start_req: proto.StartJobRequest,
    exit_proc_fut: asyncio.Event,
    cch: utils.aio.duplex_unix._AsyncDuplex,
    inference_client: _InferenceRunnerClient,
) -> JobTask:
    # used to warn users if none of connect/shutdown is called inside the job_entry
    ctx_connect, ctx_shutdown = False, False
    room = rtc.Room()
    request_shutdown_fut = asyncio.Future[_ShutdownInfo]()

    @room.on("disconnected")
    def _on_room_disconnected(*args):
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
    room._info.name = info.job.room.name
    job_ctx = JobContext(
        proc=proc,
        info=info,
        room=room,
        on_connect=_on_ctx_connect,
        on_shutdown=_on_ctx_shutdown,
        inference_executor=inference_client,
    )

    @utils.log_exceptions(logger=logger)
    async def _run_job_task() -> None:
        utils.http_context._new_session_ctx()
        job_ctx_token = _JobContextVar.set(job_ctx)

        job_entry_task = asyncio.create_task(
            job_entrypoint_fnc(job_ctx), name="job_entrypoint"
        )

        async def _warn_not_connected_task():
            await asyncio.sleep(10)
            if not ctx_connect and not ctx_shutdown:
                logger.warning(
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
                logger.warning("job task completed without connecting or shutting down")

        job_entry_task.add_done_callback(log_exception)

        shutdown_info = await request_shutdown_fut
        logger.debug(
            "shutting down job task",
            extra={
                "reason": shutdown_info.reason,
                "user_initiated": shutdown_info.user_initiated,
            },
        )
        await channel.asend_message(cch, proto.Exiting(reason=shutdown_info.reason))
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
        _JobContextVar.reset(job_ctx_token)
        exit_proc_fut.set()

    task = asyncio.create_task(_run_job_task())
    job_task = JobTask(job_ctx=job_ctx, task=task, shutdown_fut=request_shutdown_fut)
    return job_task


async def _async_main(
    proc: JobProcess,
    job_entrypoint_fnc: Callable[[JobContext], Any],
    mp_cch: socket.socket,
) -> None:
    cch = await duplex_unix._AsyncDuplex.open(mp_cch)

    job_task: JobTask | None = None
    exit_proc_fut = asyncio.Event()
    no_msg_timeout = utils.aio.sleep(proto.PING_INTERVAL * 5)  # missing 5 pings

    inference_client = _InferenceRunnerClient(cch=cch)

    @utils.log_exceptions(logger=logger)
    async def _read_ipc_task():
        nonlocal job_task
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

            if isinstance(msg, proto.StartJobRequest):
                assert job_task is None, "job task already running"
                job_task = _start_job(
                    proc, job_entrypoint_fnc, msg, exit_proc_fut, cch, inference_client
                )

            if isinstance(msg, proto.ShutdownRequest):
                if job_task is None:
                    break  # there is no running job, we can exit immediately

                with contextlib.suppress(asyncio.InvalidStateError):
                    job_task.shutdown_fut.set_result(
                        _ShutdownInfo(reason=msg.reason, user_initiated=False)
                    )

            if isinstance(msg, proto.InferenceResponse):
                inference_client._on_inference_response(msg)

    async def _self_health_check():
        await no_msg_timeout
        print("worker process is not responding.. worker crashed?")
        with contextlib.suppress(asyncio.CancelledError):
            exit_proc_fut.set()

    read_task = asyncio.create_task(_read_ipc_task(), name="ipc_read")
    health_check_task = asyncio.create_task(_self_health_check(), name="health_check")

    def _done_cb(task: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            exit_proc_fut.set()

    read_task.add_done_callback(_done_cb)

    await exit_proc_fut.wait()
    await utils.aio.gracefully_cancel(read_task, health_check_task)

    with contextlib.suppress(duplex_unix.DuplexClosed):
        await cch.aclose()


@dataclass
class ProcStartArgs:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Any]
    log_cch: socket.socket
    mp_cch: socket.socket
    asyncio_debug: bool
    user_arguments: Any | None = None
    inference_runners: dict[str, Callable[[], InferenceExecutor]] | None = None


@dataclass
class ThreadStartArgs:
    mp_cch: socket.socket
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Any]
    user_arguments: Any | None
    asyncio_debug: bool
    join_fnc: Callable[[], None]


def thread_main(
    args: ThreadStartArgs,
) -> None:
    """main function for the job process when using the ThreadedJobRunner"""
    tid = threading.get_native_id()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(args.asyncio_debug)
    loop.slow_callback_duration = 0.1  # 100ms

    cch = duplex_unix._Duplex.open(args.mp_cch)
    try:
        init_req = channel.recv_message(cch, proto.IPC_MESSAGES)
        assert isinstance(
            init_req, proto.InitializeRequest
        ), "first message must be InitializeRequest"
        job_proc = JobProcess(start_arguments=args.user_arguments)

        logger.debug("initializing job runner", extra={"tid": tid})
        args.initialize_process_fnc(job_proc)
        logger.debug("job runner initialized", extra={"tid": tid})
        channel.send_message(cch, proto.InitializeResponse())

        main_task = loop.create_task(
            _async_main(job_proc, args.job_entrypoint_fnc, cch.detach()),
            name="job_proc_main",
        )
        loop.run_until_complete(main_task)
    except duplex_unix.DuplexClosed:
        pass
    except Exception:
        logger.exception("error while running job process", extra={"tid": tid})
    finally:
        args.join_fnc()
        loop.run_until_complete(loop.shutdown_default_executor())
