from __future__ import annotations

import argparse
import asyncio
import contextvars
import os
import signal
import sys
import threading
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal

from ..job import JobExecutorType
from ..log import logger
from ..voice import AgentSession, io
from ..voice.transcription import TranscriptSynchronizer
from ..worker import AgentServer, WorkerOptions
from . import proto
from .log import setup_logging

if TYPE_CHECKING:
    from ..voice.remote_session import TcpSessionTransport
    from .tcp_console import TcpAudioInput, TcpAudioOutput

HANDLED_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
)


class _ExitCli(BaseException):
    pass


ConsoleMode = Literal["text", "audio"]


class AgentsConsole:
    """Minimal console stub for TCP console mode (Go CLI handles the TUI)."""

    _instance: AgentsConsole | None = None

    @classmethod
    def get_instance(cls) -> AgentsConsole:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import datetime
        import pathlib

        self._lock = threading.Lock()
        self._io_acquired = False
        self._io_acquired_event = threading.Event()
        self._enabled = False
        self._record = False
        self._console_mode: ConsoleMode = "audio"
        self._tcp_transport: TcpSessionTransport | None = None
        self._tcp_audio_input: TcpAudioInput | None = None
        self._tcp_audio_output: TcpAudioOutput | None = None
        self._session_directory = pathlib.Path(
            "console-recordings",
            f"session-{datetime.datetime.now().strftime('%m-%d-%H%M%S')}",
        )

    def acquire_io(self, *, loop: asyncio.AbstractEventLoop, session: AgentSession) -> None:
        with self._lock:
            if self._io_acquired:
                raise RuntimeError("the ConsoleIO was already acquired by another session")

            if asyncio.get_running_loop() != loop:
                raise RuntimeError(
                    "the ConsoleIO must be acquired in the same asyncio loop as the session"
                )

            self._io_acquired = True
            self._io_loop = loop
            self._io_context = contextvars.copy_context()

            assert self._tcp_transport is not None
            assert self._tcp_audio_input is not None
            assert self._tcp_audio_output is not None
            self._io_audio_input = self._tcp_audio_input
            self._io_audio_output = self._tcp_audio_output

            self._io_transcription_sync = TranscriptSynchronizer(
                next_in_chain_audio=self._io_audio_output,
                next_in_chain_text=None,
            )
            self._io_acquired_event.set()
            self._io_session = session

        if session:
            if self._tcp_transport is not None:
                session._session_transport = self._tcp_transport
                session._session_transport_audio_input = self._tcp_audio_input
                session._session_transport_audio_output = self._tcp_audio_output

            self._update_sess_io(
                session,
                self.console_mode,
                self._io_audio_input,
                self._io_transcription_sync.audio_output,
                self._io_transcription_sync.text_output,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, val: bool) -> None:
        self._record = val

    @property
    def session_directory(self) -> Any:
        return self._session_directory

    @property
    def io_acquired(self) -> bool:
        with self._lock:
            return self._io_acquired

    @property
    def io_session(self) -> AgentSession:
        if not self._io_acquired:
            raise RuntimeError("AgentsConsole is not acquired")
        return self._io_session

    @property
    def io_loop(self) -> asyncio.AbstractEventLoop:
        if not self._io_acquired:
            raise RuntimeError("AgentsConsole is not acquired")
        return self._io_loop

    @property
    def io_context(self) -> contextvars.Context:
        if not self._io_acquired:
            raise RuntimeError("AgentsConsole is not acquired")
        return self._io_context

    def wait_for_io_acquisition(self) -> None:
        self._io_acquired_event.wait()

    @property
    def console_mode(self) -> ConsoleMode:
        return self._console_mode

    @console_mode.setter
    def console_mode(self, mode: ConsoleMode) -> None:
        with self._lock:
            self._console_mode = mode

            if not self._io_acquired:
                return

            self.io_loop.call_soon_threadsafe(
                self._update_sess_io,
                self.io_session,
                mode,
                self._io_audio_input,
                self._io_transcription_sync.audio_output,
                self._io_transcription_sync.text_output,
            )

    def _update_sess_io(
        self,
        sess: AgentSession,
        mode: ConsoleMode,
        audio_input: io.AudioInput,
        audio_output: io.AudioOutput,
        text_output: io.TextOutput,
    ) -> None:
        if asyncio.get_running_loop() != self.io_loop:
            raise RuntimeError("_update_sess_io must be executed on the io_loop")

        with self._lock:
            if not self._io_acquired:
                return

            if self._io_session != sess or self._console_mode != mode:
                return

            if mode == "text":
                sess.input.audio = None
                sess.output.audio = None
                sess.output.transcription = None
            else:
                sess.input.audio = audio_input
                sess.output.audio = audio_output
                sess.output.transcription = text_output


def _run_tcp_console(*, server: AgentServer, connect_addr: str, record: bool = False) -> None:
    """Run console in TCP mode — connects to the Go CLI's TCP server."""
    from ..voice.remote_session import TcpSessionTransport
    from .tcp_console import TcpAudioInput, TcpAudioOutput

    host, port_str = connect_addr.rsplit(":", 1)
    port = int(port_str)

    setup_logging("DEBUG", devmode=True, console=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tcp_audio_input: TcpAudioInput | None = None

    async def _tcp_main() -> None:
        nonlocal tcp_audio_input
        transport = TcpSessionTransport(host, port)

        server._job_executor_type = JobExecutorType.THREAD

        console_inst = AgentsConsole.get_instance()
        console_inst.enabled = True
        console_inst.record = record
        console_inst._tcp_transport = transport
        tcp_audio_input = TcpAudioInput()
        console_inst._tcp_audio_input = tcp_audio_input
        console_inst._tcp_audio_output = TcpAudioOutput(transport)

        @server.once("worker_started")
        def _simulate_job() -> None:
            asyncio.run_coroutine_threadsafe(
                server.simulate_job(
                    "console-room", agent_identity="console", fake_job=True
                ),
                loop,
            )

        try:
            await server.run(devmode=True, unregistered=True)
        finally:
            await transport.close()

    exit_triggered = False

    async def _graceful_shutdown() -> None:
        if tcp_audio_input is not None:
            tcp_audio_input.close()
        await server.aclose()

    def _handle_exit(sig: int, frame: FrameType | None) -> None:
        nonlocal exit_triggered
        if exit_triggered:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        exit_triggered = True
        asyncio.run_coroutine_threadsafe(_graceful_shutdown(), loop)

    for sig in HANDLED_SIGNALS:
        signal.signal(sig, _handle_exit)

    try:
        loop.run_until_complete(_tcp_main())
    finally:
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, lambda *_: os._exit(1))

        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except Exception:
            pass
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            loop.close()


def _run_worker(server: AgentServer, args: proto.CliArgs) -> None:
    devmode = args.reload_addr is not None

    exit_raised = False

    def _handle_exit(sig: int, frame: FrameType | None) -> None:
        nonlocal exit_raised
        if exit_raised:
            os._exit(1)
        exit_raised = True
        raise _ExitCli()

    for sig in HANDLED_SIGNALS:
        signal.signal(sig, _handle_exit)

    setup_logging(args.log_level, devmode=devmode, console=False)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.slow_callback_duration = 0.1  # 100ms

    async def _worker_run(worker: AgentServer) -> None:
        try:
            await server.run(devmode=devmode, unregistered=False)
        except Exception:
            logger.exception("worker failed")

    watch_client = None
    if args.reload_addr:
        from .watcher import WatchClient

        watch_client = WatchClient(server, args.reload_addr, loop=loop)
        watch_client.start()

    try:
        main_task = loop.create_task(_worker_run(server), name="worker_main_task_cli")
        try:
            loop.run_until_complete(main_task)
        except _ExitCli:
            pass

        # Second Ctrl+C force-exits.
        def _force_exit(sig: int, frame: FrameType | None) -> None:
            logger.warning("exiting forcefully", extra={"signal": sig})
            os._exit(1)

        for sig in HANDLED_SIGNALS:
            signal.signal(sig, _force_exit)

        try:
            if not devmode:
                loop.run_until_complete(server.drain())
            loop.run_until_complete(server.aclose())

            if watch_client:
                loop.run_until_complete(watch_client.aclose())
        except _ExitCli:
            pass  # stray from first signal — ignore
    finally:
        # Re-enable force exit for the final cleanup phase
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, lambda *_: os._exit(1))

        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except Exception:
            pass
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            loop.close()


def run_app(server: AgentServer | WorkerOptions) -> None:
    if isinstance(server, WorkerOptions):
        server = AgentServer.from_server_options(server)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    start_p = sub.add_parser("start")
    start_p.add_argument("--log-level", default="INFO")
    start_p.add_argument("--url")
    start_p.add_argument("--api-key")
    start_p.add_argument("--api-secret")
    start_p.add_argument("--reload-addr")

    console_p = sub.add_parser("console")
    console_p.add_argument("--connect-addr", required=True)
    console_p.add_argument("--record", action="store_true", default=False)

    args = parser.parse_args()
    if args.command is None:
        print("Please use the Go CLI: lk agent start|dev|console")
        sys.exit(1)

    if args.command == "console":
        _run_tcp_console(server=server, connect_addr=args.connect_addr, record=args.record)
    elif args.command == "start":
        _run_worker(
            server=server,
            args=proto.CliArgs(
                log_level=args.log_level,
                url=args.url,
                api_key=args.api_key,
                api_secret=args.api_secret,
                reload_addr=args.reload_addr,
            ),
        )
