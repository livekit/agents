"""Python launcher adapter: runs the supplied server, normalizes logs, and owns asyncio."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from builtins import ExceptionGroup
from collections.abc import Mapping

from livekit.protocol import agent

from . import _core
from ._bindings import CLIError
from ._generated.cli import (
    CliArgs,
    CliArgsConsole,
    CliArgsStart,
    CliParseResultDisplay,
    LogRecord,
)
from .job import JobAcceptArguments, JobExecutorType, RunningJobInfo
from .log import DEV_LEVEL, _add_global_log_fields, logger
from .types import NOT_GIVEN
from .worker import AgentServer, Plugin

__all__ = ["run_app"]

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


def run_app(server: AgentServer) -> None:
    """Launch an explicitly supplied server through the same command interface."""
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


# --- logs --------------------------------------------------------------------------------------


# `logging.LogRecord`'s own attributes, which a line never repeats as extras.
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
    """A record's attribute as JSON, as the reference's encoder writes it: a date by its
    ``isoformat()``, anything else not JSON by its ``str()``."""
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
    """Render JSON, escaping Unicode only when the host stream requires it."""

    def __init__(self, *, ensure_ascii: bool) -> None:
        super().__init__()
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        line = _core.format_log_record(_log_record(record, self))
        if self._ensure_ascii:
            return json.dumps(json.loads(line), ensure_ascii=True, separators=(",", ":"))
        return line


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
_log_config: str | None = None


def _python_log_level(level: str) -> int:
    aliases = {"WARN": "WARNING", "FATAL": "CRITICAL"}
    name = aliases.get(level.upper(), level.upper())
    if name == "OFF":
        return logging.CRITICAL + 1
    if name == "TRACE":
        logging.addLevelName(5, "TRACE")
        return 5
    try:
        return logging.getLevelNamesMapping()[name]
    except KeyError:
        raise ValueError(f"unknown log level: {level}") from None


def setup_logging(log_level: str) -> logging.Handler:
    """Write structured JSON logs to inherited stdout at the requested level."""
    global _log_config
    _log_config = log_level
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="backslashreplace")
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("_", "-")
    handler = _LineHandler()
    handler.setFormatter(_LogFormatter(ensure_ascii=encoding not in ("utf-8", "utf8")))
    _add_global_log_fields(handler)
    root = logging.getLogger()
    root.addHandler(handler)
    level = _python_log_level(log_level)
    root.setLevel(level)
    for name in NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        if noisy.level == logging.NOTSET:
            noisy.setLevel(logging.WARN)
    if logger.level == logging.NOTSET:
        logger.setLevel(level)

    def _plugin_level(plugin: Plugin) -> None:
        if plugin.logger is not None and plugin.logger.level == logging.NOTSET:
            plugin.logger.setLevel(level)

    for plugin in Plugin.registered_plugins:
        _plugin_level(plugin)
    Plugin.emitter.on("plugin_registered", _plugin_level)
    return handler


# --- the worker --------------------------------------------------------------------------------


def _run_worker(server: AgentServer, args: CliArgsStart, *, unregistered: bool = False) -> int:
    """Serve, stop admission, drain, and clean up within one process deadline."""
    try:
        server._update_options(
            ws_url=args.ws_url if args.ws_url is not None else NOT_GIVEN,
            api_key=args.api_key if args.api_key is not None else NOT_GIVEN,
            api_secret=args.api_secret if args.api_secret is not None else NOT_GIVEN,
            shutdown_timeout=args.shutdown_timeout
            if args.shutdown_timeout is not None
            else NOT_GIVEN,
            drain_timeout=args.drain_timeout if args.drain_timeout is not None else NOT_GIVEN,
        )
    except ValueError as configuration_error:
        raise CLIError(str(configuration_error)) from configuration_error
    server._config.simulation = args.simulation
    devmode = args.dev
    setup_logging(args.log_level)

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
            # the reason leads: a launcher shows the message, the traceback only on request
            logger.error("worker failed: %s", error, exc_info=error)
    finally:
        _close_loop(loop, shutdown)
    return int(error is not None)


async def _wait_for_exit(main_task: asyncio.Task[None], shutdown: _core.Shutdown) -> None:
    # Construct the native awaitable inside the running host loop.
    exit_task = asyncio.ensure_future(shutdown.wait())
    try:
        await asyncio.wait({main_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        exit_task.cancel()


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
    task = asyncio.create_task(server._drain())
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if not done:
        task.cancel()
        logger.warning("drain timed out, closing remaining work")
        return
    try:
        task.result()
    except asyncio.TimeoutError:
        logger.warning("drain timed out, closing remaining work")


def _close_loop(loop: asyncio.AbstractEventLoop, shutdown: _core.Shutdown) -> None:
    shutdown.begin()
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.run_until_complete(loop.shutdown_default_executor())
    loop.close()
    shutdown.closed()
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
            for info in self._server._active_jobs
        ]

    async def _core_reload_jobs(
        self, jobs: list[tuple[bytes, str, str, str, str, str, str, bool]]
    ) -> None:
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
    from .voice._session_bindings import ConsoleCall

    setup_logging("DEBUG")
    ConsoleCall.addr, ConsoleCall.record = connect_addr, record
    server._update_options(job_executor_type=JobExecutorType.THREAD)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown = _core.Shutdown(server._config.shutdown_timeout, 0.0)

    async def _call() -> None:
        await server._simulate_job("console-room", agent_identity="console", fake_job=True)
        # the console is one call: the process exits once its job ended
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
