"""What runs inside a job process (or a job thread): the handshake with the
parent and its messages over the core's ``IpcClient``, the process's own log lines, and the job
itself: its ``JobContext``, the steps of the shutdown the core runs (``_core.JobRun``), its spans.
The parent's side is ``ipc/__init__.py``."""

from __future__ import annotations

from multiprocessing import current_process

if current_process().name == "job_proc":
    import signal

    # the parent handles the signals (and asks the process to shut down)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: _dump_stack_traces())

import asyncio
import contextlib
import contextvars
import inspect
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, create_autospec

from opentelemetry import trace

from livekit import rtc
from livekit.protocol import agent

from .. import _core
from ..job import (
    JobAcceptArguments,
    JobContext,
    JobExecutorType,
    JobProcess,
    RunningJobInfo,
    _JobContextVar,
    callable_name,
)
from ..log import logger
from ..telemetry import trace_types, tracer, utils as trace_utils
from ..telemetry._transport import use_traceparent
from ..utils import aio, http_context, log_exceptions
from .launch import load_preload_modules

_T = TypeVar("_T")


@dataclass
class JobArgs:
    """What a job process (or thread) is started with."""

    job_entrypoint_fnc: Callable[[JobContext], Any]
    session_end_fnc: Callable[[JobContext], Awaitable[None]] | None
    simulation_end_fnc: Callable[[Any], Any] | None
    session_end_timeout: float
    preload_modules: tuple[str, ...] = ()
    setup_fnc: Callable[[JobProcess], None] | None = None
    user_arguments: Any | None = None
    mp_cch: socket.socket | None = None
    log_config: tuple[str, str] | None = None
    logger_levels: dict[str, int] = field(default_factory=dict)


def job_main(args: JobArgs) -> None:
    # a process forked from the forkserver inherits its core: this one is a job process
    _core.init(_core.Role.Job)
    job_proc = _JobProc(
        args.job_entrypoint_fnc,
        args.session_end_fnc,
        session_end_timeout=args.session_end_timeout,
        executor_type=JobExecutorType.PROCESS,
        user_arguments=args.user_arguments,
        simulation_end_fnc=args.simulation_end_fnc,
        preload_modules=args.preload_modules,
        setup_fnc=args.setup_fnc,
    )
    log_to_stdout(args.log_config, args.logger_levels, job_proc.log_fields)
    serve_parent(cast(socket.socket, args.mp_cch), job_proc)

    import sys
    import threading
    import traceback

    for t in threading.enumerate():
        if threading.main_thread() == t:
            continue

        if threading.current_thread() == t:
            continue

        if t.daemon:
            continue

        from concurrent.futures.thread import _threads_queues

        if t in _threads_queues:
            continue

        t.join(timeout=0.25)

        frames = sys._current_frames()
        frame = frames.get(t.ident)  # type: ignore

        logger.warning(
            f"non-daemon thread `{t.name}` may prevent the process from exiting",
            extra={"thread_id": t.native_id, "thread_name": t.name},
        )

        if frame is not None:
            logger.warning("stack for `%s`:\n%s", t.name, "".join(traceback.format_stack(frame)))


class _JobProc:
    """A job process's (or job thread's) job. The core decides what ends the job and runs the
    shutdown in order and within its bounds (``_core.JobRun``); the ``_core_*`` methods are the
    steps it runs, each in the span it opened."""

    def __init__(
        self,
        job_entrypoint_fnc: Callable[[JobContext], Any],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        *,
        session_end_timeout: float,
        executor_type: JobExecutorType,
        user_arguments: Any | None = None,
        simulation_end_fnc: Callable[[Any], Any] | None = None,
        preload_modules: Sequence[str] = (),
        setup_fnc: Callable[[JobProcess], None] | None = None,
    ) -> None:
        self._executor_type = executor_type
        self._user_arguments = user_arguments
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._session_end_fnc = session_end_fnc
        self._simulation_end_fnc = simulation_end_fnc
        self._preload_modules = tuple(preload_modules)
        self._setup_fnc = setup_fnc
        self._session_end_timeout = session_end_timeout
        self._job_task: asyncio.Task[None] | None = None
        # stamped on the process's log lines
        self.log_fields: dict[str, Any] = {"pid": os.getpid()}
        # the parent's socket, set once initialized
        self._client: _core.IpcClient

    def initialize(self, http_proxy: str | None) -> None:
        self._job_proc = JobProcess(
            executor_type=self._executor_type,
            user_arguments=self._user_arguments,
            http_proxy=http_proxy,
        )
        # the framework's warm-up; a no-op under a forkserver, which imported it already
        from . import _preload

        load_preload_modules(self._preload_modules)
        _preload._step("the livekit-rtc runtime", _preload._rtc_runtime)
        _preload._step("the local inference models", _preload._local_inference_models)
        if self._setup_fnc is not None:
            setup_fnc: Callable[[JobProcess], object] = self._setup_fnc
            result = setup_fnc(self._job_proc)
            if inspect.isawaitable(result) or inspect.isasyncgen(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError(
                    "setup_fnc returned an awaitable or async generator; "
                    "process setup must be synchronous"
                )

    # --- the parent's messages, as the core's IpcClient serves them -------------------------

    async def _core_start_job(self, fields: _core.StartJobFields) -> _core.JobRun:
        with use_traceparent(fields.traceparent):
            return self._start_job(
                RunningJobInfo(
                    accept_arguments=JobAcceptArguments(
                        name=fields.name, identity=fields.identity, metadata=fields.metadata
                    ),
                    job=agent.Job.FromString(bytes(fields.job)),
                    url=fields.url,
                    token=fields.token,
                    worker_id=fields.worker_id,
                    fake_job=fields.fake_job,
                    received_at=fields.received_at,
                    accepted_at=fields.accepted_at,
                    assigned_at=fields.assigned_at,
                    launched_at=fields.launched_at,
                )
            )

    async def _core_dump_stacks(self) -> None:
        _dump_stack_traces()

    async def _core_ping(self) -> None:
        """Answered once the loop ran it: the pong measures the loop's responsiveness."""

    async def aclose(self) -> None:
        # the parent went or the job ended: a job still running is cancelled
        if self._job_task is not None:
            await aio.cancel_and_wait(self._job_task)

    def _start_job(self, info: RunningJobInfo) -> _core.JobRun:
        job = info.job
        self.log_fields.update(job_id=job.id, room=job.room.name, room_id=job.room.sid)
        if info.fake_job:
            self._room = cast(rtc.Room, _mock_room())
        else:
            self._room = rtc.Room()
        self._room._info.name = info.job.room.name

        self._run = _core.JobRun(
            fake_job=info.fake_job,
            session_end_timeout=self._session_end_timeout,
            redaction=info.job.enable_redaction,
            url=info.url,
            recording=info.job.enable_recording,
            session_end=callable_name(self._session_end_fnc) if self._session_end_fnc else None,
        )
        self._room.on("disconnected", lambda *_: self._run.shutdown("room disconnected", False))
        self._job_ctx = JobContext(
            proc=self._job_proc,
            info=info,
            room=self._room,
            run=self._run,
        )
        # Reachable from the remote session port (via get_job_context) when a simulation finalizes.
        self._job_ctx._simulation_end_fnc = self._simulation_end_fnc

        self._job_task = asyncio.create_task(self._run_job_task(), name="job_task")
        self._job_task.add_done_callback(lambda _: self._client.stop())
        return self._run

    @log_exceptions(logger=logger)
    async def _run_job_task(self) -> None:
        self._job_ctx._on_setup()
        self._job_ctx._start_log_buffering()
        # before the first span: without a provider the job's root would not record
        self._job_ctx._prepare_telemetry()

        job_ctx_token = _JobContextVar.set(self._job_ctx)
        http_context._new_session_ctx()
        # the core calls the steps below from its own tasks: user code gets the job's context
        # (get_job_context() and its http session) back
        self._http_session = http_context._ContextVar.get()

        # the job's root span, from the availability request to the end of shutdown; the
        # entrypoint returning is an event on it
        self._job_span = _start_job_span(self._job_ctx)
        self._shutdown_span: trace.Span | None = None
        try:
            await self._run.run(self)
        finally:
            # whatever happened above (a cancel while waiting, a close that raised), the job
            # ends: root span, temp dir, per-job telemetry state
            if self._shutdown_span is not None:
                self._shutdown_span.end()
            self._job_span.end()

            await self._run.close_tasks()

            await self._job_ctx._on_cleanup()
            await http_context._close_http_ctx()
            _JobContextVar.reset(job_ctx_token)

    # --- the steps the core runs -------------------------------------------------------

    def _in_job(self, awaitable: Awaitable[_T], name: str) -> asyncio.Task[_T]:
        ctx = contextvars.copy_context()
        ctx.run(_JobContextVar.set, self._job_ctx)
        ctx.run(http_context._ContextVar.set, self._http_session)

        async def run() -> _T:
            return await awaitable

        return ctx.run(asyncio.create_task, run(), name=name)

    async def _core_start_entrypoint(self) -> asyncio.Task[None]:
        return self._in_job(self._traceable_entrypoint(), "job_user_entrypoint")

    async def _traceable_entrypoint(self) -> None:
        job_span = self._job_span
        with tracer.use_span(
            job_span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            try:
                await self._job_entrypoint_fnc(self._job_ctx)
            except asyncio.CancelledError:
                job_span.add_event("entrypoint_cancelled")
                raise
            except Exception as e:
                trace_utils.record_exception(job_span, e)
                raise
            job_span.add_event("entrypoint_returned")

    async def _core_wait(self, task: asyncio.Task[None]) -> None:
        await asyncio.shield(task)

    async def _core_cancel(self, task: asyncio.Task[None]) -> None:
        await aio.cancel_and_wait(task)

    async def _core_shutdown_began(self, reason: str, user_initiated: bool) -> None:
        self._shutdown_span = tracer.start_span(
            "job_shutdown",
            context=trace.set_span_in_context(self._job_span),
            attributes={
                trace_types.ATTR_SHUTDOWN_REASON: reason,
                trace_types.ATTR_SHUTDOWN_USER_INITIATED: user_initiated,
            },
        )

    def _core_span(self, name: str, callback: str | None) -> trace.Span:
        attributes = {trace_types.ATTR_CALLBACK_NAME: callback} if callback else None
        return tracer.start_span(
            name, context=trace.set_span_in_context(self._shutdown_span), attributes=attributes
        )

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None:
        logger.log(level, msg, extra=extra, exc_info=exc)

    def _core_span_end(
        self, span: trace.Span, error: BaseException | None, timed_out: bool
    ) -> None:
        if error is not None or timed_out:
            trace_utils.record_exception(span, error or asyncio.TimeoutError())
        span.end()

    async def _core_close_session(self) -> None:
        if session := self._job_ctx._primary_agent_session:
            await self._in_job(session.aclose(), "job_close_session")

    async def _core_session_end(self, span: trace.Span) -> None:
        assert self._session_end_fnc is not None
        with trace.use_span(span, end_on_exit=False, record_exception=False):
            await self._in_job(self._session_end_fnc(self._job_ctx), "job_session_end")

    async def _core_upload(self, span: trace.Span) -> None:
        with trace.use_span(span, end_on_exit=False, record_exception=False):
            await self._in_job(self._job_ctx._on_session_end(), "job_upload")

    async def _core_disconnect(self, span: trace.Span) -> None:
        with trace.use_span(span, end_on_exit=False, record_exception=False):
            await self._room.disconnect()

    async def _core_shutdown_callback(
        self,
        callback: Callable[..., Awaitable[None]],
        reason: str | None,
        span: trace.Span | None,
    ) -> None:
        args = () if reason is None else (reason,)
        # a framework callback runs in the shutdown's own span
        with trace.use_span(
            span or self._shutdown_span or trace.INVALID_SPAN,
            end_on_exit=False,
            record_exception=False,
        ):
            await self._in_job(_awaited(callback, args), "job_shutdown_callback")


async def _awaited(callback: Callable[..., Awaitable[None]], args: tuple[str, ...]) -> None:
    # a callback may return any awaitable (a lambda returning a future)
    await callback(*args)


def _start_job_span(job_ctx: JobContext) -> trace.Span:
    """The job's root span, ``job_entrypoint``, back-dated to the availability request; ended
    by ``_run_job_task`` after the shutdown sequence."""
    job = job_ctx.job
    info = job_ctx._info
    entrypoint_started_at = time.time()
    start_time_ns = int(info.received_at * 1e9) if info.received_at else None
    span = tracer.start_span(
        "job_entrypoint",
        start_time=start_time_ns,
        attributes={
            trace_types.ATTR_JOB_ID: job.id,
            trace_types.ATTR_AGENT_NAME: job.agent_name,
            trace_types.ATTR_ROOM_NAME: job.room.name,
            trace_types.ATTR_ROOM_SID: job.room.sid,
            trace_types.ATTR_DISPATCH_ID: job.dispatch_id,
            trace_types.ATTR_WORKER_ID: info.worker_id,
            trace_types.ATTR_JOB_AGENT_ID: job.state.agent_id,
        },
    )
    events, attributes = _core.job_timeline(
        info.received_at,
        info.accepted_at,
        info.assigned_at,
        info.launched_at,
        entrypoint_started_at,
        job.state.started_at,
    )
    for name, timestamp in events:
        span.add_event(name, timestamp=timestamp)
    span.set_attributes(dict(attributes))
    return span


def thread_main(args: JobArgs) -> None:
    """A job thread's main: the job runs on a loop of the thread's own, logging as the worker."""
    job_proc = _JobProc(
        args.job_entrypoint_fnc,
        args.session_end_fnc,
        session_end_timeout=args.session_end_timeout,
        executor_type=JobExecutorType.THREAD,
        user_arguments=args.user_arguments,
        simulation_end_fnc=args.simulation_end_fnc,
        preload_modules=args.preload_modules,
        setup_fnc=args.setup_fnc,
    )
    serve_parent(cast(socket.socket, args.mp_cch), job_proc)


def log_to_stdout(
    config: tuple[str, str] | None, levels: dict[str, int], fields: dict[str, Any]
) -> None:
    """A child process writes its own log lines, as the worker does (``config``: its level and
    line style), to the stdout it inherits; each record stamped with ``fields``, the process's
    (its pid, the job's ids once one runs). A worker that set up no logging (``server._run()``
    from the user's code) has its processes log at its root level, colored on a terminal."""
    import sys

    from ..cli import setup_logging

    if config is None:
        level = logging.getLevelName(levels.get("root", logging.WARNING))
        config = (level, "dev" if sys.stdout.isatty() else "json")
    setup_logging(*config).addFilter(_Stamp(fields))
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


class _Stamp(logging.Filter):
    def __init__(self, fields: dict[str, Any]) -> None:
        super().__init__()
        self._fields = fields

    def filter(self, record: logging.LogRecord) -> bool:
        # a record's own extra wins
        for key, value in self._fields.items():
            record.__dict__.setdefault(key, value)
        return True


def serve_parent(mp_cch: socket.socket, host: Any) -> None:
    """A child process's (or job thread's) side of the parent's socket: the handshake
    (``host.initialize(http_proxy)``), then a loop of its own on which ``_core.IpcClient.serve``
    hands the parent's messages to ``host`` until the parent goes or the child is done. A failed
    initialization raises after notifying the parent."""
    _initialize_stack_dump_worker()
    client = _core.IpcClient(
        mp_cch.detach(), host.initialize, process=host._executor_type == JobExecutorType.PROCESS
    )
    host._client = client

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(client.asyncio_debug)
    loop.slow_callback_duration = 0.1  # 100ms

    try:
        task = loop.create_task(_serve(client, host), name="proc_client_main")
        while not task.done():
            # the parent handles the process's shutdown (its ShutdownRequest)
            with contextlib.suppress(KeyboardInterrupt):
                loop.run_until_complete(task)
    finally:
        # shut the loop down the way asyncio.run() does, a thread runner reuses the process
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@log_exceptions(logger=logger)
async def _serve(client: _core.IpcClient, host: Any) -> None:
    try:
        await client.serve(host)
    finally:
        await host.aclose()
        await client.aclose()


_stack_dump_worker: ThreadPoolExecutor | None = None
_stack_dump_busy = threading.Lock()


def _initialize_stack_dump_worker() -> None:
    global _stack_dump_worker
    if os.getenv("LK_DUMP_STACK_TRACES", "0").lower() in ("0", "false", "no"):
        return
    if _stack_dump_worker is None:
        # Warm the single writer before the job event loop exists.
        _stack_dump_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stack-dump")
        _stack_dump_worker.submit(lambda: None).result()


def _dump_stack_traces() -> None:
    """Take loop-affine references; the bounded writer does all rendering and filesystem I/O."""
    if os.getenv("LK_DUMP_STACK_TRACES", "0").lower() in ("0", "false", "no"):
        return
    worker = _stack_dump_worker
    if worker is None or not _stack_dump_busy.acquire(blocking=False):
        return
    try:
        frames = tuple(sys._current_frames().items())
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            tasks: tuple[tuple[str, bool, bool, tuple[FrameType, ...]], ...] = ()
        else:
            tasks = tuple(
                (task.get_name(), task.done(), task.cancelled(), tuple(task.get_stack(limit=64)))
                for task in asyncio.all_tasks(loop)
            )
        worker.submit(_write_stack_dump, frames, tasks)
    except BaseException:
        _stack_dump_busy.release()
        raise


def _write_stack_dump(
    frames: tuple[tuple[int, FrameType], ...],
    tasks: tuple[tuple[str, bool, bool, tuple[FrameType, ...]], ...],
) -> None:
    import tempfile
    import traceback
    from pathlib import Path

    import psutil

    try:
        directory = os.getenv("LK_DUMP_DIR", tempfile.gettempdir())
        Path(directory).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=directory,
            delete=False,
            prefix=f"livekit-agents-pid-{os.getpid()}-{time.time_ns()}-",
            suffix=".stacktrace",
        ) as output:
            print(f"Process {current_process().name} (pid {os.getpid()})", file=output)
            for thread_id, frame in frames:
                print(f"\nThread {thread_id}", file=output)
                traceback.print_stack(frame, limit=64, file=output)
            print(f"\nASYNCIO TASKS ({len(tasks)})", file=output)
            for name, done, cancelled, stack in tasks:
                print(f"\nTask {name}: done={done}, cancelled={cancelled}", file=output)
                for frame in stack:
                    traceback.print_stack(frame, limit=1, file=output)
            try:
                process = psutil.Process()
                print(f"\nMemory: {process.memory_info()}", file=output)
                print(f"Threads: {process.num_threads()}", file=output)
            except psutil.Error as error:
                print(f"Memory sample unavailable: {error}", file=output)
        logger.info("wrote stack trace dump", extra={"path": output.name})
    except Exception:
        logger.exception("could not write stack trace dump")
    finally:
        _stack_dump_busy.release()


def _mock_room() -> Any:
    MockRoom = create_autospec(rtc.Room, instance=True)
    MockRoom.local_participant = create_autospec(rtc.LocalParticipant, instance=True)
    # autospec leaves attributes as truthy mocks; pin sid/identity to real strings
    # so they don't leak into places like inference request headers (see _utils.py).
    MockRoom.local_participant.sid = ""
    MockRoom.local_participant.identity = "agent"
    MockRoom._info = create_autospec(rtc.room.proto_room.RoomInfo, instance=True)  # type: ignore
    MockRoom.isconnected.return_value = True
    MockRoom.name = "console"
    MockRoom.metadata = ""
    MockRoom.num_participants = 2
    MockRoom.num_publishers = 2
    MockRoom.connection_state = rtc.ConnectionState.CONN_CONNECTED
    MockRoom.departure_timeout = 0
    MockRoom.empty_timeout = 0

    MockRoom.sid = AsyncMock(return_value="RM_mock_sid")

    mock_remote_participant = create_autospec(rtc.RemoteParticipant, instance=True)
    mock_remote_participant.identity = "mock_user"
    mock_remote_participant.sid = "PA_mock_user"
    mock_remote_participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    mock_remote_participant.state = rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE
    MockRoom.remote_participants = {mock_remote_participant.identity: mock_remote_participant}
    return MockRoom
