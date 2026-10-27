"""What runs inside a job or inference process (or a job thread): the handshake with the
parent and its messages over the core's ``IpcClient``, the process's own log lines, and the job
itself: its ``JobContext``, the steps of the shutdown the core runs (``_core.JobRun``), its spans.
The parent's side is ``ipc/__init__.py``."""

from __future__ import annotations

from multiprocessing import current_process

if current_process().name in ("job_proc", "agents_inference_process"):
    import signal

    # the parent handles the signals (and asks the process to shut down)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: _dump_stack_traces())

import asyncio
import contextlib
import contextvars
import functools
import logging
import math
import os
import socket
import time
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, create_autospec

from opentelemetry import trace

from livekit import rtc

from .. import _core
from ..inference_runner import _RunnersDict
from ..job import JobContext, JobExecutorType, JobProcess, RunningJobInfo, _JobContextVar
from ..log import logger
from ..telemetry import loop_monitor, trace_types, tracer, utils as trace_utils
from ..utils import aio, http_context, log_exceptions

_T = TypeVar("_T")


@dataclass
class JobArgs:
    """What a job process (or thread) is started with."""

    job_entrypoint_fnc: Callable[[JobContext], Any]
    session_end_fnc: Callable[[JobContext], Awaitable[None]] | None
    simulation_end_fnc: Callable[[Any], Any] | None
    session_end_timeout: float
    user_arguments: Any | None = None
    mp_cch: socket.socket | None = None
    log_config: tuple[str, str] | None = None
    logger_levels: dict[str, int] = field(default_factory=dict)


@dataclass
class InferenceArgs:
    mp_cch: socket.socket
    runners: _RunnersDict
    log_config: tuple[str, str] | None


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
    )
    log_to_stdout(args.log_config, args.logger_levels, job_proc.log_fields)
    serve_parent(cast(socket.socket, args.mp_cch), job_proc, job=True)

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
    ) -> None:
        self._executor_type = executor_type
        self._user_arguments = user_arguments
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._session_end_fnc = session_end_fnc
        self._simulation_end_fnc = simulation_end_fnc
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
        from . import _preload  # noqa: F401

    # --- the parent's messages, as the core's IpcClient serves them -------------------------

    async def _core_start_job(self, fields: dict[str, Any]) -> _core.JobRun:
        return self._start_job(RunningJobInfo._from_wire(fields))

    async def _core_dump_stacks(self) -> None:
        _dump_stack_traces()

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
            session_end=self._session_end_fnc,
        )
        self._room.on("disconnected", lambda *_: self._run.shutdown("room disconnected", False))
        self._job_ctx = JobContext(
            proc=self._job_proc,
            info=info,
            room=self._room,
            run=self._run,
            inference_executor=self._client,
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

    def _in_job(self, coro: Coroutine[Any, Any, _T], name: str) -> asyncio.Task[_T]:
        ctx = contextvars.copy_context()
        ctx.run(_JobContextVar.set, self._job_ctx)
        ctx.run(http_context._ContextVar.set, self._http_session)
        return ctx.run(asyncio.create_task, coro, name=name)

    async def _core_start_entrypoint(self) -> asyncio.Task[None]:
        return self._in_job(self._traceable_entrypoint(), "job_user_entrypoint")

    async def _traceable_entrypoint(self) -> None:
        job_span = self._job_span
        with tracer.use_span(
            job_span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            # the loop monitor's heartbeat predates the job: give its reports this context
            if (monitor := loop_monitor.get_monitor(asyncio.get_running_loop())) is not None:
                monitor.set_report_context(contextvars.copy_context())
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
        args: tuple[str, ...],
        span: trace.Span | None,
    ) -> None:
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
    )
    serve_parent(cast(socket.socket, args.mp_cch), job_proc, job=True)


def inference_main(args: InferenceArgs) -> None:
    _core.init(_core.Role.Inference)
    log_to_stdout(args.log_config, {}, {"pid": os.getpid(), "inference": True})
    serve_parent(args.mp_cch, _InferenceProc(args.runners), job=False)


class _InferenceProc:
    def __init__(self, runners: _RunnersDict) -> None:
        # create an instance of each runner (the ctor must not requires any argument)
        self._runners = {name: runner() for name, runner in runners.items()}
        self._executor = ThreadPoolExecutor(max_workers=math.ceil(_core.cpu_count()))

    def initialize(self, http_proxy: str | None) -> None:
        for runner in self._runners.values():
            try:
                logger.debug(
                    "initializing inference runner",
                    extra={"runner": runner.__class__.INFERENCE_METHOD},
                )
                start_time = time.perf_counter()
                runner.initialize()
                logger.debug(
                    "inference runner initialized",
                    extra={
                        "runner": runner.__class__.INFERENCE_METHOD,
                        "elapsed_time": time.perf_counter() - start_time,
                    },
                )
            except Exception:
                logger.exception(
                    "error initializing inference runner",
                    extra={"runner": runner.__class__.INFERENCE_METHOD},
                )

    # --- the parent's messages, as the core's IpcClient serves them -------------------------

    async def _core_dump_stacks(self) -> None:
        _dump_stack_traces()

    async def _core_inference(self, method: str, data: bytes) -> bytes | None:
        if method not in self._runners:
            logger.warning("unknown inference method", extra={"method": method})
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._runners[method].run, data)

    async def aclose(self) -> None:
        pass


def log_to_stdout(
    config: tuple[str, str] | None, levels: dict[str, int], fields: dict[str, Any]
) -> None:
    """A child process writes its own log lines, as the worker does (``config``: its level and
    line style), to the stdout it inherits; each record stamped with ``fields``, the process's
    (its pid, the job's ids once one runs). A worker that set up no logging (``server.run()``
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


def serve_parent(mp_cch: socket.socket, host: Any, *, job: bool) -> None:
    """A child process's (or job thread's) side of the parent's socket: the handshake
    (``host.initialize(http_proxy)``), then a loop of its own on which ``_core.IpcClient.serve``
    hands the parent's messages to ``host`` until the parent goes or the child is done. A failed
    initialization returns at once (the parent was told)."""
    try:
        client = _core.IpcClient(mp_cch.detach(), host.initialize, job=job)
    except Exception:
        return
    host._client = client

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(client.asyncio_debug)
    loop.slow_callback_duration = 0.1  # 100ms
    # flag synchronous code blocking the job loop (spans + warnings), see telemetry.loop_monitor
    loop_monitor.start_monitoring(loop, name="job")

    try:
        task = loop.create_task(_serve(client, host), name="proc_client_main")
        while not task.done():
            # the parent handles the process's shutdown (its ShutdownRequest)
            with contextlib.suppress(KeyboardInterrupt):
                loop.run_until_complete(task)
    finally:
        loop_monitor.stop_monitoring(loop)
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


def _dump_stack_traces() -> None:
    """The process's threads and asyncio tasks, and its memory, in a file under
    ``LK_DUMP_DIR`` (the temp dir), when ``LK_DUMP_STACK_TRACES`` is set."""
    import asyncio
    import faulthandler
    import os
    import tempfile
    import time
    import traceback
    from multiprocessing import current_process
    from pathlib import Path

    import psutil

    if os.getenv("LK_DUMP_STACK_TRACES", "0").lower() in ("0", "false", "no"):
        return

    dir: str = os.getenv("LK_DUMP_DIR", tempfile.gettempdir())
    Path(dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=dir,
        delete=False,
        prefix=f"livekit-agents-pid-{current_process().pid}-{time.time_ns()}-",
        suffix=".stacktrace",
    ) as f:
        print(f"\n{'=' * 60}", file=f)
        print(
            f"Process {current_process().name} (pid {current_process().pid}) stack trace dump",
            file=f,
        )
        print(f"{'=' * 60}\n", file=f)

        faulthandler.dump_traceback(file=f, all_threads=True)
        print("\n", file=f)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                print("=" * 60, file=f)
                print("ASYNCIO TASKS", file=f)
                print("=" * 60, file=f)

                tasks = asyncio.all_tasks(loop)
                print(f"Total tasks: {len(tasks)}\n", file=f)

                for i, task in enumerate(tasks, 1):
                    print(f"\n--- Task {i}/{len(tasks)} ---", file=f)
                    print(f"Name: {task.get_name()}", file=f)
                    print(f"Done: {task.done()}", file=f)

                    if not task.done():
                        print(f"Cancelled: {task.cancelled()}", file=f)

                        try:
                            stack = task.get_stack()
                            print(f"Stack frames: {len(stack)}", file=f)
                            print("Stack trace:", file=f)
                            for frame in stack:
                                traceback.print_stack(frame, limit=1, file=f)
                        except Exception as e:
                            print(f"Could not get stack: {e}", file=f)

                        try:
                            coro = task.get_coro()
                            print(f"Coroutine: {coro}", file=f)
                            if cr_frame := getattr(coro, "cr_frame", None):
                                print("Coroutine frame:", file=f)
                                traceback.print_stack(cr_frame, file=f)
                        except Exception as e:
                            print(f"Could not get coroutine: {e}", file=f)
                    else:
                        try:
                            exc = task.exception()
                            if exc:
                                print(f"Exception: {exc}", file=f)
                                print("Exception traceback:", file=f)
                                traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
                        except Exception as e:
                            print(f"Could not get exception: {e}", file=f)

                    print("", file=f)
            else:
                print("No asyncio event loop running", file=f)
        except Exception as e:
            print(f"Error dumping asyncio tasks: {e}", file=f)
            traceback.print_exc(file=f)

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            print("\n" + "=" * 60, file=f)
            print("MEMORY USAGE", file=f)
            print("=" * 60, file=f)
            print(f"RSS: {memory_mb:.2f} MB", file=f)
            print(f"VMS: {memory_info.vms / (1024 * 1024):.2f} MB", file=f)
        except Exception:
            pass


@functools.cache
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
