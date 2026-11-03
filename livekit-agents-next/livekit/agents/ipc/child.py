"""What runs inside a job process (or a job thread): the handshake with the
parent and its messages over the core's ``IpcClient``, the process's own log lines, and the job
itself: its ``JobContext`` and the steps of the shutdown the core runs (``_core.JobRun``).
The parent's side is ``ipc/__init__.py``."""

from __future__ import annotations

from multiprocessing import current_process

if current_process().name == "job_proc":
    import signal

    # the parent handles the signals (and asks the process to shut down)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

import asyncio
import contextlib
import contextvars
import logging
import os
import socket
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from livekit import rtc
from livekit.protocol import agent

from .. import _core, _telemetry
from .._generated.worker_ipc import StartJobFields
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
    user_arguments: Any | None = None
    mp_cch: socket.socket | PipeEnd | None = None
    log_config: str | None = None
    logger_levels: dict[str, int] = field(default_factory=dict)
    # the server's settings as environment variables, which the job's code reads
    env: tuple[tuple[str, str], ...] = ()


def job_main(args: JobArgs) -> None:
    # a process forked from the forkserver inherits its core: this one is a job process
    _core.init(_core.Role.Job)
    os.environ.update(args.env)
    job_proc = _JobProc(
        args.job_entrypoint_fnc,
        args.session_end_fnc,
        session_end_timeout=args.session_end_timeout,
        executor_type=JobExecutorType.PROCESS,
        user_arguments=args.user_arguments,
        simulation_end_fnc=args.simulation_end_fnc,
        preload_modules=args.preload_modules,
        server_env=args.env,
    )
    log_to_stdout(args.log_config, args.logger_levels, job_proc.log_fields)
    serve_parent(cast("socket.socket | PipeEnd", args.mp_cch), job_proc)

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
        server_env: Sequence[tuple[str, str]] = (),
    ) -> None:
        self._executor_type = executor_type
        self._server_env = dict(server_env)
        self._user_arguments = user_arguments
        self._job_entrypoint_fnc = job_entrypoint_fnc
        self._session_end_fnc = session_end_fnc
        self._simulation_end_fnc = simulation_end_fnc
        self._preload_modules = tuple(preload_modules)
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
        # after the preloads: an application installing its own tracer provider keeps it
        _telemetry.install()
        _preload._step("the livekit-rtc runtime", _preload._rtc_runtime)
        _preload._step("the local inference models", _preload._local_inference_models)

    # --- the parent's messages, as the core's IpcClient serves them -------------------------

    async def _core_start_job(self, fields: StartJobFields) -> _core.JobRun:
        run = self._start_job(
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
        # the job's root span, before its task can run
        run.start_trace(fields)
        self._job_task = asyncio.create_task(self._run_job_task(), name="job_task")
        self._job_task.add_done_callback(lambda _: self._client.stop())
        return run

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
            observability_url=os.environ.get("LIVEKIT_OBSERVABILITY_URL"),
            api_key=self._server_env.get("LIVEKIT_API_KEY") or "",
            api_secret=self._server_env.get("LIVEKIT_API_SECRET") or "",
        )
        self._room.on("disconnected", lambda *_: self._run.shutdown("room disconnected", False))
        self._job_ctx = JobContext(
            proc=self._job_proc,
            info=info,
            room=self._room,
            run=self._run,
            server_env=self._server_env,
        )
        # Reachable from the remote session port (via get_job_context) when a simulation finalizes.
        self._job_ctx._simulation_end_fnc = self._simulation_end_fnc

        return self._run

    @log_exceptions(logger=logger)
    async def _run_job_task(self) -> None:
        self._job_ctx._on_setup()
        self._job_ctx._start_log_buffering()

        job_ctx_token = _JobContextVar.set(self._job_ctx)
        http_context._new_session_ctx()
        # the core calls the steps below from its own tasks: user code gets the job's context
        # (get_job_context() and its http session) back
        self._http_session = http_context._ContextVar.get()

        try:
            # the core records the job's spans and runs each step in its span
            await self._run.run(self)
        finally:
            # whatever happened above (a cancel while waiting, a close that raised), the job
            # ends: temp dir, per-job telemetry state
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
        return self._in_job(self._job_entrypoint_fnc(self._job_ctx), "job_user_entrypoint")

    async def _core_wait(self, task: asyncio.Task[None]) -> None:
        await asyncio.shield(task)

    async def _core_cancel(self, task: asyncio.Task[None]) -> None:
        await aio.cancel_and_wait(task)

    def _core_log(
        self, level: int, msg: str, extra: dict[str, Any], exc: BaseException | None = None
    ) -> None:
        logger.log(level, msg, extra=extra, exc_info=exc)

    async def _core_close_session(self) -> None:
        if session := self._job_ctx._primary_agent_session:
            await self._in_job(session.aclose(), "job_close_session")

    async def _core_session_end(self) -> None:
        assert self._session_end_fnc is not None
        await self._in_job(self._session_end_fnc(self._job_ctx), "job_session_end")

    async def _core_upload(self) -> None:
        await self._in_job(self._job_ctx._on_session_end(), "job_upload")

    async def _core_disconnect(self) -> None:
        await self._room.disconnect()

    async def _core_shutdown_callback(
        self, callback: Callable[..., Awaitable[None]], reason: str | None
    ) -> None:
        args = () if reason is None else (reason,)
        await self._in_job(_awaited(callback, args), "job_shutdown_callback")


async def _awaited(callback: Callable[..., Awaitable[None]], args: tuple[str, ...]) -> None:
    # a callback may return any awaitable (a lambda returning a future)
    await callback(*args)


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
        server_env=args.env,
    )
    serve_parent(cast("socket.socket | PipeEnd", args.mp_cch), job_proc)


def log_to_stdout(config: str | None, levels: dict[str, int], fields: dict[str, Any]) -> None:
    """Write the child's structured logs with the worker's level and process fields."""
    from ..cli import setup_logging

    configured_level = (
        config if config is not None else logging.getLevelName(levels.get("root", logging.WARNING))
    )
    setup_logging(configured_level).addFilter(_Stamp(fields))
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


class PipeEnd:
    """The job's end of the parent's named pipe (Windows), duplicated straight into the job
    process when multiprocessing pickles its arguments."""

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    def __reduce__(self) -> tuple[type[PipeEnd], tuple[int]]:
        from multiprocessing.context import get_spawning_popen

        popen: Any = get_spawning_popen()
        return (PipeEnd, (popen.duplicate_for_child(self._handle),))

    def dup(self) -> PipeEnd:
        if sys.platform != "win32":
            raise NotImplementedError("a pipe end exists only on Windows")
        import _winapi

        process = _winapi.GetCurrentProcess()
        return PipeEnd(
            _winapi.DuplicateHandle(
                process, self._handle, process, 0, False, _winapi.DUPLICATE_SAME_ACCESS
            )
        )

    def detach(self) -> int:
        handle, self._handle = self._handle, None
        assert handle is not None, "the pipe end was already detached"
        return handle

    def close(self) -> None:
        if sys.platform == "win32" and self._handle is not None:
            import _winapi

            _winapi.CloseHandle(self.detach())


def serve_parent(mp_cch: socket.socket | PipeEnd, host: Any) -> None:
    """A child process's (or job thread's) side of the parent's socket: the handshake
    (``host.initialize(http_proxy)``), then a loop of its own on which ``_core.IpcClient.serve``
    hands the parent's messages to ``host`` until the parent goes or the child is done. A failed
    initialization raises after notifying the parent."""
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


def _mock_room() -> Any:
    # only a fake job (the console) builds one; production never imports unittest
    from unittest.mock import AsyncMock, create_autospec

    MockRoom = create_autospec(rtc.Room, instance=True)
    MockRoom.local_participant = create_autospec(rtc.LocalParticipant, instance=True)
    # autospec leaves attributes as truthy mocks; pin sid/identity/kind to real values
    # so they don't leak into places like inference request headers (see _utils.py).
    MockRoom.local_participant.sid = ""
    MockRoom.local_participant.identity = "agent"
    MockRoom.local_participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
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
