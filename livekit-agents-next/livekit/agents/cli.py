"""``python myagent.py start|console`` and ``python -m livekit.agents``: the subprocess interface
``lk agent dev|start|console`` drives. The core parses the command line, finds the entrypoint's
module, writes the log lines, speaks the dev channel and decides what a signal means; this module
runs the event loop and imports the user's code."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import pkgutil
import signal
import sys
from builtins import ExceptionGroup
from types import FrameType
from typing import Any

from livekit.protocol import agent

from . import _core
from ._exceptions import CLIError
from ._generated.worker_cli import CliArgs, CliArgsConsole, CliArgsDownloadFiles, CliArgsStart
from .job import JobAcceptArguments, JobExecutorType, RunningJobInfo
from .log import DEV_LEVEL, _add_global_log_fields, logger
from .telemetry import loop_monitor
from .types import NOT_GIVEN
from .worker import AgentServer, Plugin

__all__ = ["main", "run_app"]

HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

# noisy loggers are set to warn by default
NOISY_LOGGERS = [
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "openai",
    "watchfiles",
    "anthropic",
    "websockets.client",
    "aiohttp.access",
    "livekit",
    "botocore",
    "aiobotocore",
    "urllib3.connectionpool",
    "mcp.client",
]


class _ExitCli(SystemExit):
    # SystemExit rather than BaseException: raised from the signal path, it can land inside
    # whatever asyncio task or callback the loop is executing, and Task.__step and Handle._run
    # re-raise only SystemExit/KeyboardInterrupt out of the loop (#5856, #6724).
    pass


def main(argv: list[str] | None = None) -> int:
    """``python -m livekit.agents``: the user's server found from the entrypoint."""
    args = _core.parse_cli(sys.argv[1:] if argv is None else argv)
    if isinstance(args, CliArgsDownloadFiles):
        return _download_files()
    return _dispatch(_load_server(args.entrypoint), args)


def run_app(server: AgentServer) -> None:
    """``python myagent.py start [--dev]``: the interface ``lk agent`` drives."""
    if status := _dispatch(server, _core.parse_cli(sys.argv[1:])):
        raise SystemExit(status)


def _dispatch(server: AgentServer, args: CliArgs) -> int:
    if isinstance(args, CliArgsConsole):
        _run_console(server, args.connect_addr, args.record)
    elif isinstance(args, CliArgsStart):
        return _run_worker(server, args)
    else:
        return _download_files()
    return 0


def _load_server(entrypoint: str | None) -> AgentServer:
    """The AgentServer the entrypoint's module defines: an ``app``, ``server`` or ``agent``
    global first, else the first one found."""
    module, path = _core.entrypoint_module(entrypoint)
    sys.path.insert(0, path)
    mod = importlib.import_module(module)
    for name in ("app", "server", "agent", *dir(mod)):
        obj = getattr(mod, name, None)
        if isinstance(obj, AgentServer):
            return obj
    raise CLIError("Could not find AgentServer in module, try to define the `server` variable")


def _download_files() -> int:
    """Import every installed ``livekit.plugins`` package and run its ``download_files``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        import livekit.plugins as plugins
    except ImportError:
        logger.warning("no livekit.plugins namespace found — nothing to download")
        return 0
    names = [m.name for m in pkgutil.iter_modules(plugins.__path__, "livekit.plugins.") if m.ispkg]
    failed = 0
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:
            logger.warning("failed to import %s: %s", name, e)
            failed = 1
    logger.info("discovered %d plugin package(s): %s", len(names), ", ".join(names) or "(none)")
    for plugin in Plugin.registered_plugins:
        logger.info("downloading files for %s", plugin.package)
        try:
            plugin.download_files()
        except Exception as e:
            logger.error("failed downloading files for %s: %s", plugin.package, e)
            failed = 1
        else:
            logger.info("finished downloading files for %s", plugin.package)
    return failed


# --- logs --------------------------------------------------------------------------------------


class _LogFormatter(logging.Formatter):
    """A record as the core writes its line (``style``: ``json``, ``dev``, ``compact`` or
    ``console``)."""

    def __init__(self, style: str, *, ensure_ascii: bool) -> None:
        super().__init__(datefmt=None if style in ("json", "dev") else "%H:%M:%S")
        self._line_style = style
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        return _core.format_log_record(
            self._line_style,
            record,
            record.getMessage(),
            self.formatTime(record, self.datefmt),
            self.formatException(record.exc_info) if record.exc_info else record.exc_text,
            self.formatStack(record.stack_info) if record.stack_info else None,
            self._ensure_ascii,
        )


class _LineHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes each record to stdout in one ``write(2)``: the job processes share the worker's
    stdout, and lines written whole do not interleave (atomically up to the pipe's buffer, and
    in practice beyond it on Linux and macOS for a single write)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            line = (self.format(record) + self.terminator).encode(encoding, "backslashreplace")
            try:
                fd = self.stream.fileno()
            except (AttributeError, OSError, ValueError):
                # not a file (a captured stdout): written as a stream
                self.stream.write(line.decode(encoding))
                self.flush()
                return
            self.flush()  # what print() buffered goes first
            while line:
                line = line[os.write(fd, line) :]
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


# what the worker logs with, for the job processes it starts: each writes its own lines
_log_config: tuple[str, str] | None = None


def setup_logging(log_level: str, style: str) -> logging.Handler:
    """Log to stdout in ``style``, at ``log_level`` for the agents' and the plugins' loggers. A
    job process logs the same way, to the stdout it inherits; returns the handler."""
    global _log_config
    _log_config = (log_level, style)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="backslashreplace")
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("_", "-")
    handler = _LineHandler()
    handler.setFormatter(_LogFormatter(style, ensure_ascii=encoding not in ("utf-8", "utf8")))
    _add_global_log_fields(handler)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(log_level)
    for name in NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        if noisy.level == logging.NOTSET:
            noisy.setLevel(logging.WARN)
    if logger.level == logging.NOTSET:
        logger.setLevel(log_level)

    def _plugin_level(plugin: Plugin) -> None:
        if plugin.logger is not None and plugin.logger.level == logging.NOTSET:
            plugin.logger.setLevel(log_level)

    for plugin in Plugin.registered_plugins:
        _plugin_level(plugin)
    Plugin.emitter.on("plugin_registered", _plugin_level)
    return handler


# --- the worker --------------------------------------------------------------------------------


def _run_worker(server: AgentServer, args: CliArgsStart, *, unregistered: bool = False) -> int:
    """``start``: the worker until a signal, then drained (not in dev) and closed; nonzero
    when the worker failed."""
    server.update_options(
        ws_url=args.ws_url if args.ws_url is not None else NOT_GIVEN,
        api_key=args.api_key if args.api_key is not None else NOT_GIVEN,
        api_secret=args.api_secret if args.api_secret is not None else NOT_GIVEN,
    )
    server._config.simulation = args.simulation
    devmode = args.dev
    setup_logging(args.log_level, args.log_style)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.slow_callback_duration = 0.1  # 100ms
    # a blocked worker loop delays job dispatch; there is no session to attach a span to here
    loop_monitor.start_monitoring(loop, name="worker", emit_spans=False)
    loop.set_exception_handler(_loop_exception_handler)

    exit_fut: asyncio.Future[None] = loop.create_future()
    exits = _core.ExitSignals()

    def _serve_exit() -> None:
        exits.served()
        if not exit_fut.done():
            exit_fut.set_result(None)

    def _on_signal(sig: int, frame: FrameType | None) -> None:
        # the first signal only schedules the exit on the loop; the core raises it in the
        # blocking frame if the loop does not serve it, and forces it on a second signal
        action = exits.signal(exit_fut.done())
        if action == "schedule":
            loop.call_soon_threadsafe(_serve_exit)
        elif action == "raise":
            raise _ExitCli()
        elif action == "force":
            logger.warning("exiting forcefully", extra={"signal": sig})
            os._exit(1)

    for sig in HANDLED_SIGNALS:
        signal.signal(sig, _on_signal)

    try:
        main_task = loop.create_task(
            _serve(server, devmode, args.cli_addr, unregistered=unregistered),
            name="worker_main_task_cli",
        )
        with contextlib.suppress(_ExitCli):
            # the worker keeps running during drain: an exit ends this wait, not the worker
            loop.run_until_complete(
                asyncio.wait({main_task, exit_fut}, return_when=asyncio.FIRST_COMPLETED)
            )
        exits.draining()
        with contextlib.suppress(_ExitCli):
            if not devmode:
                loop.run_until_complete(_drain(server))
            loop.run_until_complete(server.aclose())
    finally:
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, lambda *_: os._exit(1))
        _close_loop(loop, exits)

    if main_task.cancelled():
        return 0
    error = main_task.exception()
    if error is not None:
        logger.error("worker failed", exc_info=error)
    return int(error is not None)


async def _serve(
    server: AgentServer, devmode: bool, cli_addr: str | None, *, unregistered: bool = False
) -> None:
    """The worker, beside the dev channel to the ``lk`` CLI at ``cli_addr`` when given."""
    channel = None
    if cli_addr:
        config = server._config
        dev = _core.dev_channel(
            cli_addr,
            config.agent_name,
            config.ws_url,
            config.api_key,
            config.api_secret,
            _DevServer(server),
        )
        channel = asyncio.ensure_future(dev)
    try:
        await server.run(devmode=devmode, unregistered=unregistered)
    finally:
        if channel is not None:
            channel.cancel()


async def _drain(server: AgentServer) -> None:
    try:
        await server.drain()
    except asyncio.TimeoutError:
        logger.warning("drain timed out, forcing shutdown")


def _loop_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    # a task preempted by the exit escalation: the shutdown is already running
    if not isinstance(context.get("exception"), _ExitCli):
        loop.default_exception_handler(context)


def _close_loop(loop: asyncio.AbstractEventLoop, exits: _core.ExitSignals | None = None) -> None:
    # The native process owner remains responsive even when a task or executor holds the loop.
    exits = exits or _core.ExitSignals(escalate=False)
    exits.closing()
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop_monitor.stop_monitoring(loop)
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.run_until_complete(loop.shutdown_default_executor())
    loop.close()
    exits.closed()
    failures = [result for result in results if isinstance(result, Exception)]
    if failures:
        raise ExceptionGroup("event-loop task cleanup failed", failures)


class _DevServer:
    """The worker as the core's dev channel sees it: its running jobs as the protocol's
    fields."""

    def __init__(self, server: AgentServer) -> None:
        self._server = server

    def _core_running_jobs(self) -> list[tuple[bytes, str, str, str, str, str, str, bool]]:
        return [
            (
                info.job.SerializeToString(),
                info.accept_arguments.name,
                info.accept_arguments.identity,
                info.accept_arguments.metadata,
                info.url,
                info.token,
                info.worker_id,
                info.fake_job,
            )
            for info in self._server.active_jobs
        ]

    async def _core_reload_jobs(self, jobs: list[tuple[Any, ...]]) -> None:
        """The jobs a dev reload carried over, on this server and a fresh token: relaunched."""
        for job, name, identity, meta, url, token, worker_id, fake_job in jobs:
            info = RunningJobInfo(
                accept_arguments=JobAcceptArguments(name=name, identity=identity, metadata=meta),
                job=agent.Job.FromString(bytes(job)),
                url=url,
                token=token,
                worker_id=worker_id,
                fake_job=fake_job,
            )
            extra = {"job_id": info.job.id, "agent_name": info.job.agent_name}
            logger.log(DEV_LEVEL, "reloading job", extra=extra)
            assert self._server._jobs is not None
            await self._server._jobs.launch(info)


# --- the console -------------------------------------------------------------------------------


def _run_console(server: AgentServer, connect_addr: str, record: bool) -> None:
    """``console --connect-addr``: one call over the ``lk`` CLI's socket, run as a fake job in
    this process."""
    from .voice.remote_session import ConsoleCall

    setup_logging("DEBUG", "console")
    ConsoleCall.addr, ConsoleCall.record = connect_addr, record
    server.update_options(job_executor_type=JobExecutorType.THREAD)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    @server.once("worker_started")
    def _start_call() -> None:
        asyncio.run_coroutine_threadsafe(
            server.simulate_job("console-room", agent_identity="console", fake_job=True), loop
        )

    exits = _core.ExitSignals(escalate=False)

    def _on_signal(sig: int, frame: FrameType | None) -> None:
        if exits.signal(False) == "force":
            os._exit(1)
        asyncio.run_coroutine_threadsafe(server.aclose(), loop)

    for sig in HANDLED_SIGNALS:
        signal.signal(sig, _on_signal)

    try:
        loop.run_until_complete(server.run(devmode=True, unregistered=True))
    finally:
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, lambda *_: os._exit(1))
        _close_loop(loop, exits)
