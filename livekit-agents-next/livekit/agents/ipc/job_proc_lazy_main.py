from __future__ import annotations

from multiprocessing import current_process

if current_process().name == "job_proc":
    import signal

    # ignore signals in the jobs process (the parent process will handle them)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if hasattr(signal, "SIGUSR1"):
        from .proc_client import _dump_stack_traces

        signal.signal(signal.SIGUSR1, _dump_stack_traces)

import asyncio
import contextvars
import socket
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from opentelemetry import trace

from livekit import rtc

from .. import _core
from ..job import JobContext, JobExecutorType, JobProcess, RunningJobInfo, _JobContextVar
from ..log import _add_global_log_fields, logger
from ..telemetry import loop_monitor, trace_types, tracer, utils as trace_utils
from ..utils import aio, http_context, log_exceptions
from .proc_client import _dump_stack_traces_impl, serve_parent
from .proto import StartJobRequest

# The core's bound on VoiceSession.aclose() during the job shutdown, a guardrail against close
# paths that hang: past it the rest of the shutdown (on_session_end, ShuttingDown,
# room.disconnect, the shutdown callbacks) runs anyway.
_SESSION_ACLOSE_TIMEOUT = 60.0

_T = TypeVar("_T")


@dataclass
class ProcStartArgs:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Any]
    session_end_fnc: Callable[[JobContext], Awaitable[None]] | None
    session_end_timeout: float
    user_arguments: Any | None
    mp_cch: socket.socket
    log_cch: socket.socket
    logger_levels: dict[str, int]
    simulation_end_fnc: Callable[[Any], Any] | None = None


def proc_main(args: ProcStartArgs) -> None:
    import logging

    from .log_queue import LogQueueHandler

    root_logger = logging.getLogger()
    for name, level in args.logger_levels.items():
        logging.getLogger(name).setLevel(level)

    log_cch = aio.duplex_unix._Duplex.open(args.log_cch)
    log_handler = LogQueueHandler(log_cch)
    _add_global_log_fields(log_handler)
    root_logger.addHandler(log_handler)

    job_proc = _JobProc(
        args.initialize_process_fnc,
        args.job_entrypoint_fnc,
        args.session_end_fnc,
        session_end_timeout=args.session_end_timeout,
        executor_type=JobExecutorType.PROCESS,
        user_arguments=args.user_arguments,
        simulation_end_fnc=args.simulation_end_fnc,
    )

    serve_parent(args.mp_cch, job_proc, job=True)

    import sys
    import threading
    import traceback

    for t in threading.enumerate():
        if threading.main_thread() == t:
            continue

        if threading.current_thread() == t:
            continue

        if t == log_handler.thread:
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

    log_handler.close()


class _JobProc:
    """A job process's (or job thread's) job. The core decides what ends the job and runs the
    shutdown in order and within its bounds (``_core.JobRun``); the ``_core_*`` methods are the
    steps it runs, each in the span it opened."""

    def __init__(
        self,
        initialize_process_fnc: Callable[[JobProcess], Any],
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
        self._initialize_process_fnc = initialize_process_fnc
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._session_end_fnc = session_end_fnc
        self._simulation_end_fnc = simulation_end_fnc
        self._session_end_timeout = session_end_timeout
        self._job_task: asyncio.Task[None] | None = None
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

        self._initialize_process_fnc(self._job_proc)

    # --- the parent's messages, as the core's IpcClient serves them -------------------------

    async def _core_start_job(self, fields: dict[str, Any]) -> _core.JobRun:
        return self._start_job(StartJobRequest._from_wire(fields))

    async def _core_dump_stacks(self) -> None:
        _dump_stack_traces_impl()

    async def aclose(self) -> None:
        # the parent went or the job ended: a job still running is cancelled
        if self._job_task is not None:
            await aio.cancel_and_wait(self._job_task)

    def _start_job(self, msg: StartJobRequest) -> _core.JobRun:
        info = msg.running_job
        if info.fake_job:
            from .mock_room import create_mock_room

            self._room = cast(rtc.Room, create_mock_room())
        else:
            self._room = rtc.Room()
        self._room._info.name = info.job.room.name

        self._run = _core.JobRun(
            fake_job=info.fake_job,
            session_end_timeout=self._session_end_timeout,
            redaction=info.job.enable_redaction,
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

    def _core_span(self, name: str, callback: Callable[..., Any] | None) -> trace.Span:
        fnc = callback or (self._session_end_fnc if name == "on_session_end" else None)
        attributes = {trace_types.ATTR_CALLBACK_NAME: _callback_name(fnc)} if fnc else None
        return tracer.start_span(
            name, context=trace.set_span_in_context(self._shutdown_span), attributes=attributes
        )

    def _core_span_end(
        self, span: trace.Span, error: BaseException | None, timed_out: bool
    ) -> None:
        if error is not None or timed_out:
            trace_utils.record_exception(span, error or asyncio.TimeoutError())
        span.end()

    @property
    def _session_aclose_timeout(self) -> float:
        return _SESSION_ACLOSE_TIMEOUT

    async def _core_close_session(self) -> None:
        if session := self._job_ctx._primary_agent_session:
            await self._in_job(session.aclose(), "job_close_session")

    def _core_has_session_end(self) -> bool:
        return self._session_end_fnc is not None

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

    def _core_shutdown_callbacks(self) -> list[tuple[Callable[[str], Awaitable[None]], bool]]:
        # framework callbacks (the session's own close) get no span of their own
        return [(cb, _is_framework_callback(cb)) for cb in self._job_ctx._shutdown_callbacks]

    async def _core_shutdown_callback(
        self, callback: Callable[[str], Awaitable[None]], reason: str, span: trace.Span | None
    ) -> None:
        # a framework callback runs in the shutdown's own span
        with trace.use_span(
            span or self._shutdown_span or trace.INVALID_SPAN,
            end_on_exit=False,
            record_exception=False,
        ):
            await self._in_job(callback(reason), "job_shutdown_callback")


def _is_framework_callback(fnc: Any) -> bool:
    """Registered by livekit-agents itself rather than by the user's entrypoint."""
    module = getattr(fnc, "__module__", None) or ""
    return module == "livekit.agents" or module.startswith("livekit.agents.")


def _callback_name(fnc: Any) -> str:
    return str(getattr(fnc, "__qualname__", None) or getattr(fnc, "__name__", None) or repr(fnc))


def _server_timestamp_seconds(value: int) -> float:
    """``JobState`` timestamps are int64; the server writes unix nanoseconds. Be tolerant
    of milliseconds/seconds should that ever change."""
    if value > 1e17:
        return value / 1e9
    if value > 1e11:
        return value / 1e3
    return float(value)


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
    _record_dispatch_timeline(span, info, entrypoint_started_at)
    return span


def _record_dispatch_timeline(
    span: trace.Span, info: RunningJobInfo, entrypoint_started_at: float
) -> None:
    """Stamp the dispatch stages on ``span``: one timestamped event per stage instant, and
    the seconds between adjacent stages as attributes (they sum to the dispatch latency).

    Timestamps travel from the worker through ``StartJobRequest``; a zero means the stage
    is unknown (simulation, console, resumed job) and is skipped rather than guessed."""
    stages = [
        ("job_received", info.received_at),
        ("job_accepted", info.accepted_at),
        ("job_assigned", info.assigned_at),
        ("process_assigned", info.launched_at),
        ("entrypoint_started", entrypoint_started_at),
    ]
    for event_name, ts in stages:
        if ts:
            span.add_event(event_name, timestamp=int(ts * 1e9))

    def _gap(attr: str, start: float, end: float) -> None:
        if start and end:
            span.set_attribute(attr, max(end - start, 0.0))

    _gap(trace_types.ATTR_JOB_ACCEPT_LATENCY, info.received_at, info.accepted_at)
    _gap(trace_types.ATTR_JOB_ASSIGNMENT_LATENCY, info.accepted_at, info.assigned_at)
    _gap(trace_types.ATTR_JOB_LAUNCH_LATENCY, info.assigned_at, info.launched_at)
    _gap(trace_types.ATTR_JOB_ENTRYPOINT_LATENCY, info.launched_at, entrypoint_started_at)
    _gap(trace_types.ATTR_JOB_DISPATCH_LATENCY, info.received_at, entrypoint_started_at)
    if (server_started := info.job.state.started_at) > 0:
        # the server's own start time, for lining up with server-side traces
        started = _server_timestamp_seconds(server_started)
        span.add_event("job_started_on_server", timestamp=int(started * 1e9))


@dataclass
class ThreadStartArgs:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Any]
    session_end_fnc: Callable[[JobContext], Awaitable[None]] | None
    session_end_timeout: float
    mp_cch: socket.socket
    user_arguments: Any | None
    simulation_end_fnc: Callable[[Any], Any] | None = None


def thread_main(
    args: ThreadStartArgs,
) -> None:
    """main function for the job process when using the ThreadedJobRunner"""
    job_proc = _JobProc(
        args.initialize_process_fnc,
        args.job_entrypoint_fnc,
        args.session_end_fnc,
        session_end_timeout=args.session_end_timeout,
        executor_type=JobExecutorType.THREAD,
        user_arguments=args.user_arguments,
        simulation_end_fnc=args.simulation_end_fnc,
    )

    serve_parent(args.mp_cch, job_proc, job=True)
