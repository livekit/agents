from __future__ import annotations

from multiprocessing import current_process
from types import TracebackType

if current_process().name == "job_proc":
    import signal
    import sys

    # ignore signals in the jobs process (the parent process will handle them)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def _no_traceback_excepthook(
        exc_type: type[BaseException], exc_val: BaseException, traceback: TracebackType | None
    ) -> None:
        if isinstance(exc_val, KeyboardInterrupt):
            return
        sys.__excepthook__(exc_type, exc_val, traceback)

    sys.excepthook = _no_traceback_excepthook


import asyncio
import contextlib
import socket
from dataclasses import dataclass
from typing import Any, Callable, cast

from livekit import rtc

from ..cli import cli
from ..debug import tracing
from ..job import JobContext, JobExecutorType, JobProcess, _JobContextVar
from ..log import logger
from ..utils import aio, http_context, log_exceptions, shortuuid
from .channel import Message
from .inference_executor import InferenceExecutor
from .proc_client import _ProcClient
from .proto import (
    Exiting,
    InferenceRequest,
    InferenceResponse,
    InitializeRequest,
    ShutdownRequest,
    StartJobRequest,
    TracingRequest,
    TracingResponse,
)


@dataclass
class ProcStartArgs:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Any]
    mp_cch: socket.socket
    log_cch: socket.socket
    user_arguments: Any | None = None


def proc_main(args: ProcStartArgs) -> None:
    from .proc_client import _ProcClient

    job_proc = _JobProc(
        args.initialize_process_fnc,
        args.job_entrypoint_fnc,
        JobExecutorType.PROCESS,
        args.user_arguments,
    )

    client = _ProcClient(
        args.mp_cch,
        args.log_cch,
        job_proc.initialize,
        job_proc.entrypoint,
    )

    client.initialize_logger()
    try:
        client.initialize()
    except Exception:
        return  # initialization failed, exit (initialize will send an error to the worker)
    client.run()


class _InfClient(InferenceExecutor):
    def __init__(self, proc_client: _ProcClient) -> None:
        self._client = proc_client
        self._active_requests: dict[str, asyncio.Future[InferenceResponse]] = {}

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        request_id = shortuuid("inference_job_")
        fut = asyncio.Future[InferenceResponse]()

        await self._client.send(
            InferenceRequest(request_id=request_id, method=method, data=data),
        )

        self._active_requests[request_id] = fut

        inf_resp = await fut
        if inf_resp.error:
            raise RuntimeError(f"inference of {method} failed: {inf_resp.error}")

        return inf_resp.data

    def _on_inference_response(self, resp: InferenceResponse) -> None:
        fut = self._active_requests.pop(resp.request_id, None)
        if fut is None:
            logger.warning("received unexpected inference response", extra={"resp": resp})
            return

        with contextlib.suppress(asyncio.InvalidStateError):
            fut.set_result(resp)


@dataclass
class _ShutdownInfo:
    user_initiated: bool
    reason: str


class _JobProc:
    def __init__(
        self,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Any],
        executor_type: JobExecutorType,
        user_arguments: Any | None = None,
    ) -> None:
        self._executor_type = executor_type
        self._user_arguments = user_arguments
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._job_task: asyncio.Task[None] | None = None

        # used to warn users if both connect and shutdown are not called inside the job_entry
        self._ctx_connect_called = False
        self._ctx_shutdown_called = False

    @property
    def has_running_job(self) -> bool:
        return self._job_task is not None

    def initialize(self, init_req: InitializeRequest, client: _ProcClient) -> None:
        self._client = client
        self._inf_client = _InfClient(client)
        self._job_proc = JobProcess(
            executor_type=self._executor_type,
            user_arguments=self._user_arguments,
            http_proxy=init_req.http_proxy or None,
        )
        self._initialize_process_fnc(self._job_proc)

    @log_exceptions(logger=logger)
    async def entrypoint(self, cch: aio.ChanReceiver[Message]) -> None:
        self._exit_proc_flag = asyncio.Event()
        self._shutdown_fut: asyncio.Future[_ShutdownInfo] = asyncio.Future()

        @log_exceptions(logger=logger)
        async def _read_ipc_task() -> None:
            async for msg in cch:
                if isinstance(msg, StartJobRequest):
                    if self.has_running_job:
                        logger.warning("trying to start a new job while one is already running")
                        continue

                    self._start_job(msg)
                if isinstance(msg, ShutdownRequest):
                    if not self.has_running_job:
                        self._exit_proc_flag.set()
                        break  # exit immediately

                    with contextlib.suppress(asyncio.InvalidStateError):
                        self._shutdown_fut.set_result(
                            _ShutdownInfo(reason=msg.reason, user_initiated=False)
                        )

                if isinstance(msg, InferenceResponse):
                    self._inf_client._on_inference_response(msg)

                if isinstance(msg, TracingRequest):
                    if not self.has_running_job:
                        logger.warning("tracing request received without running job")
                        return

                    try:
                        job_ctx_token = _JobContextVar.set(self._job_ctx)
                        tracing_tasks = []
                        for callback in self._job_ctx._tracing_callbacks:
                            tracing_tasks.append(
                                asyncio.create_task(callback(), name="job_tracing_callback")
                            )

                        await asyncio.gather(*tracing_tasks)
                        _JobContextVar.reset(job_ctx_token)
                    except Exception:
                        logger.exception("error while exeuting tracing tasks")

                    await self._client.send(
                        TracingResponse(
                            request_id=msg.request_id,
                            info=tracing.Tracing._get_job_handle(self._job_ctx.job.id)._export(),
                        )
                    )

        read_task = asyncio.create_task(_read_ipc_task(), name="job_ipc_read")

        await self._exit_proc_flag.wait()
        await aio.cancel_and_wait(read_task)

    def _start_job(self, msg: StartJobRequest) -> None:
        if cli.CLI_ARGUMENTS is not None and cli.CLI_ARGUMENTS.console:
            from .mock_room import MockRoom

            self._room = cast(rtc.Room, MockRoom)
        else:
            self._room = rtc.Room()

        @self._room.on("disconnected")
        def _on_room_disconnected(*args: Any) -> None:
            with contextlib.suppress(asyncio.InvalidStateError):
                self._shutdown_fut.set_result(
                    _ShutdownInfo(user_initiated=False, reason="room disconnected")
                )

        def _on_ctx_connect() -> None:
            self._ctx_connect_called = True

        def _on_ctx_shutdown(reason: str) -> None:
            self._ctx_shutdown_called = True

            with contextlib.suppress(asyncio.InvalidStateError):
                self._shutdown_fut.set_result(_ShutdownInfo(user_initiated=True, reason=reason))

        self._room._info.name = msg.running_job.job.room.name

        self._job_ctx = JobContext(
            proc=self._job_proc,
            info=msg.running_job,
            room=self._room,
            on_connect=_on_ctx_connect,
            on_shutdown=_on_ctx_shutdown,
            inference_executor=self._inf_client,
        )

        self._job_task = asyncio.create_task(self._run_job_task(), name="job_task")

        def _exit_proc_cb(_: asyncio.Task[None]) -> None:
            self._exit_proc_flag.set()

        self._job_task.add_done_callback(_exit_proc_cb)

    async def _run_job_task(self) -> None:
        job_ctx_token = _JobContextVar.set(self._job_ctx)
        http_context._new_session_ctx()

        job_entry_task = asyncio.create_task(
            self._job_entrypoint_fnc(self._job_ctx), name="job_user_entrypoint"
        )

        async def _warn_not_connected_task() -> None:
            if cli.CLI_ARGUMENTS is not None and cli.CLI_ARGUMENTS.console:
                return

            await asyncio.sleep(10)
            if not self._ctx_connect_called and not self._ctx_shutdown_called:
                logger.warning(
                    "The room connection was not established within 10 seconds after calling job_entry. "  # noqa: E501
                    "This may indicate that job_ctx.connect() was not called. "
                )

        warn_unconnected_task = asyncio.create_task(_warn_not_connected_task())
        job_entry_task.add_done_callback(lambda _: warn_unconnected_task.cancel())

        def log_exception(t: asyncio.Task[Any]) -> None:
            if not t.cancelled() and t.exception():
                logger.error(
                    "unhandled exception while running the job task",
                    exc_info=t.exception(),
                )
            elif not self._ctx_connect_called and not self._ctx_shutdown_called:
                if cli.CLI_ARGUMENTS is not None and cli.CLI_ARGUMENTS.console:
                    return

                logger.warning(
                    "The job task completed without establishing a connection or performing a proper shutdown. "  # noqa: E501
                    "Ensure that job_ctx.connect()/job_ctx.shutdown() is called and the job is correctly finalized."  # noqa: E501
                )

        job_entry_task.add_done_callback(log_exception)

        shutdown_info = await self._shutdown_fut
        logger.debug(
            "shutting down job task",
            extra={
                "reason": shutdown_info.reason,
                "user_initiated": shutdown_info.user_initiated,
            },
        )

        await self._client.send(Exiting(reason=shutdown_info.reason))
        await self._room.disconnect()

        try:
            shutdown_tasks = []
            for callback in self._job_ctx._shutdown_callbacks:
                shutdown_tasks.append(
                    asyncio.create_task(
                        callback(shutdown_info.reason), name="job_shutdown_callback"
                    )
                )

            await asyncio.gather(*shutdown_tasks)
        except Exception:
            logger.exception("error while shutting down the job")

        await http_context._close_http_ctx()
        _JobContextVar.reset(job_ctx_token)


@dataclass
class ThreadStartArgs:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Any]
    join_fnc: Callable[[], None]
    mp_cch: socket.socket
    user_arguments: Any | None


def thread_main(
    args: ThreadStartArgs,
) -> None:
    """main function for the job process when using the ThreadedJobRunner"""
    try:
        from .proc_client import _ProcClient

        job_proc = _JobProc(
            args.initialize_process_fnc,
            args.job_entrypoint_fnc,
            JobExecutorType.THREAD,
            args.user_arguments,
        )

        client = _ProcClient(
            args.mp_cch,
            None,
            job_proc.initialize,
            job_proc.entrypoint,
        )

        client.initialize()
        client.run()
    finally:
        args.join_fnc()
