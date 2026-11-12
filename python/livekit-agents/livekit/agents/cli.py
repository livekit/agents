from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from builtins import ExceptionGroup
from collections.abc import Mapping

from . import _core
from ._bindings import CLIError
from ._generated.cli import (
    CliArgs,
    CliArgsConsole,
    CliArgsStart,
    CliParseResultDisplay,
    LogLevel,
    LogRecord,
    LogStyle,
)
from ._generated.worker_options import ProcessOptions
from .http import client as http_client
from .job import JobExecutorType
from .log import _add_global_log_fields, logger
from .utils.aio import close_loop
from .worker import AgentServer

__all__ = ["run_app"]

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


def run_app(server: AgentServer) -> None:
    """Run ``server`` with the command (start, console) given on the command line."""
    http_client._export_ca_paths()
    if status := _launch(sys.argv[1:], server):
        raise SystemExit(status)


def _launch(argv: list[str], server: AgentServer) -> int:
    parsed = _core.parse_cli([sys.argv[0], *argv])
    if isinstance(parsed, CliParseResultDisplay):
        print(parsed.message, file=sys.stderr if parsed.stderr else sys.stdout, end="")
        return parsed.exit_code
    try:
        args = parsed.args
        return _dispatch(server, args)
    except CLIError as error:
        print(str(error), file=sys.stderr)
        return 2


def _dispatch(server: AgentServer, args: CliArgs) -> int:
    if isinstance(args, CliArgsConsole):
        _run_console(server, args.connect_addr, args.record)
        return 0
    return _run_worker(server, args)


_RECORD_ATTRIBUTES = frozenset(logging.makeLogRecord({}).__dict__) | {"asctime", "message"}


def _log_record(record: logging.LogRecord, formatter: logging.Formatter) -> LogRecord:
    fields = {
        key: _lenient(value)
        for key, value in record.__dict__.items()
        if key not in _RECORD_ATTRIBUTES and not key.startswith("_")
    }
    if isinstance(record.msg, dict):
        fields.update(_lenient_mapping(record.msg))
    return LogRecord(
        level=record.levelname,
        logger=record.name,
        process_id=record.process,
        timestamp=record.created,
        message=record.getMessage() if not isinstance(record.msg, dict) else "",
        fields=fields,
        exception=formatter.formatException(record.exc_info)
        if record.exc_info
        else record.exc_text,
        stack=formatter.formatStack(record.stack_info) if record.stack_info else None,
    )


def _lenient_mapping(value: Mapping[object, object]) -> dict[str, object]:
    return {str(key): _lenient(item) for key, item in value.items()}


def _lenient(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return _lenient_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_lenient(v) for v in value]
    if callable(isoformat := getattr(value, "isoformat", None)):
        return str(isoformat())
    return str(value)


class _LogFormatter(logging.Formatter):
    def __init__(self, style: LogStyle, *, ensure_ascii: bool) -> None:
        super().__init__()
        self._line_style = style
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        line = _core.format_log_record(_log_record(record, self), self._line_style)
        if self._ensure_ascii and self._line_style == "json":
            return json.dumps(json.loads(line), ensure_ascii=True, separators=(",", ":"))
        return line


class _LineHandler(logging.StreamHandler):  # type: ignore[type-arg]
    # One write(2) per record: job processes share the worker's stdout, and whole-line writes
    # do not interleave.

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            line = (self.format(record) + self.terminator).encode(encoding, "backslashreplace")
            try:
                fd = self.stream.fileno()
            except (AttributeError, OSError, ValueError):
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


# Passed to the job processes so each configures its own logging the same way.
_log_config: tuple[LogLevel, LogStyle] | None = None


def setup_logging(log_level: LogLevel, style: LogStyle = "json") -> logging.Handler:
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
    level = _core.host_log_level(log_level)
    if log_level == "trace":
        logging.addLevelName(level, "TRACE")
    root.setLevel(level)
    for name in NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        if noisy.level == logging.NOTSET:
            noisy.setLevel(logging.WARN)
    for framework in (logger, logging.getLogger("livekit.plugins")):
        if framework.level == logging.NOTSET:
            framework.setLevel(level)
    return handler


def _run_worker(server: AgentServer, args: CliArgsStart, *, unregistered: bool = False) -> int:
    try:
        server._config.update(args.options, args.process)
    except ValueError as configuration_error:
        raise CLIError(str(configuration_error)) from configuration_error
    server._config.simulation = args.simulation
    devmode = args.dev
    setup_logging(args.log_level, "dev" if devmode else "json")

    shutdown = _core.Shutdown(server._config.shutdown_timeout, server._config.drain_timeout)
    shutdown.listen()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.slow_callback_duration = 0.1
    main_task = loop.create_task(
        _serve(server, devmode, args.cli_addr, unregistered=unregistered),
        name="worker_main_task_cli",
    )
    try:
        try:
            loop.run_until_complete(_wait_for_exit(main_task, shutdown))
        finally:
            loop.run_until_complete(_finish_server(server, shutdown, drain=not devmode))
        error = main_task.exception() if main_task.done() and not main_task.cancelled() else None
        if error is not None:
            # Message first: the launcher shows it, and the traceback only on request.
            logger.error("worker failed: %s", error, exc_info=error)
    finally:
        _close_loop(loop, shutdown)
    return int(error is not None)


async def _wait_for_exit(main_task: asyncio.Task[None], shutdown: _core.Shutdown) -> None:
    # The native awaitable must be constructed inside the running loop.
    exit_task = asyncio.ensure_future(shutdown.wait())
    try:
        await asyncio.wait({main_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        exit_task.cancel()


async def _serve(
    server: AgentServer, devmode: bool, cli_addr: str | None, *, unregistered: bool = False
) -> None:
    channel = None
    if cli_addr:
        config = server._config
        dev = _core.dev_channel(
            cli_addr,
            config.agent_name,
            config.ws_url,
            config.api_key,
            config.api_secret,
            server._ensure_runtime(),
        )
        channel = asyncio.ensure_future(dev)
    try:
        await server._run(devmode=devmode, unregistered=unregistered)
    finally:
        if channel is not None:
            channel.cancel()


async def _finish_server(server: AgentServer, shutdown: _core.Shutdown, *, drain: bool) -> None:
    shutdown.begin()
    try:
        if drain:
            await _drain(server, shutdown.drain_remaining())
    finally:
        await server.aclose()


async def _drain(server: AgentServer, timeout: float) -> None:
    if timeout <= 0:
        return
    try:
        await server._drain(timeout)
    except TimeoutError:
        logger.warning("drain timed out, closing remaining work")


def _close_loop(loop: asyncio.AbstractEventLoop, shutdown: _core.Shutdown) -> None:
    shutdown.begin()
    failures = close_loop(loop)
    shutdown.closed()
    if failures:
        raise ExceptionGroup("event-loop task cleanup failed", failures)


def _run_console(server: AgentServer, connect_addr: str, record: bool) -> None:
    from .voice._session_bindings import ConsoleCall

    setup_logging("debug", "console")
    ConsoleCall.addr, ConsoleCall.record = connect_addr, record
    server._update_options(ProcessOptions(job_executor_type=JobExecutorType.THREAD))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown = _core.Shutdown(server._config.shutdown_timeout, 0.0)

    async def _call() -> None:
        await server._simulate_job("console-room", agent_identity="console", fake_job=True)
        # The console is one call: exit once its job ends.
        await server._drain(None)
        shutdown.begin()

    def _start_call() -> None:
        asyncio.run_coroutine_threadsafe(_call(), loop)

    server._on_started.append(_start_call)

    shutdown.listen()
    main_task = loop.create_task(server._run(devmode=True, unregistered=True))
    try:
        try:
            loop.run_until_complete(_wait_for_exit(main_task, shutdown))
        finally:
            loop.run_until_complete(_finish_server(server, shutdown, drain=False))
        if main_task.done() and not main_task.cancelled():
            main_task.result()
    finally:
        _close_loop(loop, shutdown)
