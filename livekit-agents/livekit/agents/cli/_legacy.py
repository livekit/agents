"""Deprecated rich Python CLI (``start``/``connect``/``download-files``).

These commands have moved to the LiveKit CLI (``lk agent ...``), which drives the
thin interface in ``cli.py`` over a subprocess. This module is kept for backwards
compatibility and will be removed in a future release; ``cli.run_app`` routes the
legacy commands here after emitting a deprecation warning.

``console`` and ``dev`` are gone: their implementations were removed and the
commands now only point at ``lk agent console`` / ``lk agent dev``.
"""

from __future__ import annotations

import asyncio
import datetime
import enum
import json
import logging
import os
import sys
import traceback
from typing import Annotated, Any

import typer
from rich.console import Console, ConsoleRenderable, Group, RenderableType
from rich.segment import Segment
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from rich.theme import Theme

from livekit import api

from .._exceptions import CLIError
from ..log import logger
from ..plugin import Plugin
from ..utils import shortuuid
from ..worker import AgentServer, ServerEnvOption, WorkerOptions
from . import cli as _cli, proto
from .log import JsonFormatter, _merge_record_extra, _silence_noisy_loggers

TRACE_LOG_LEVEL = 5


class _ExitCli(BaseException):
    pass


# from https://github.com/encode/uvicorn/blob/c1144fd4f130388cffc05ee17b08747ce8c1be11/uvicorn/importer.py#L9C1-L34C20
# def import_from_string(import_str: Any) -> Any:
#     if not isinstance(import_str, str):
#         return import_str

#     module_str, _, attrs_str = import_str.partition(":")
#     if not module_str or not attrs_str:
#         message = 'Import string "{import_str}" must be in format "<module>:<attribute>".'
#         raise RuntimeError(message.format(import_str=import_str))

#     try:
#         module = importlib.import_module(module_str)
#     except ModuleNotFoundError as exc:
#         if exc.name != module_str:
#             raise exc from None
#         message = 'Could not import module "{module_str}".'
#         raise RuntimeError(message.format(module_str=module_str)) from None

#     instance = module
#     try:
#         for attr_str in attrs_str.split("."):
#             instance = getattr(instance, attr_str)
#     except AttributeError:
#         message = 'Attribute "{attrs_str}" not found in module "{module_str}".'
#         raise RuntimeError(message.format(attrs_str=attrs_str, module_str=module_str)) from None

#     return instance


class AgentsConsole:
    _instance: AgentsConsole | None = None

    @classmethod
    def get_instance(cls) -> AgentsConsole:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        theme: dict[str, str | Style] = {
            "tag": "black on #1fd5f9",
            "label": "#8f83ff",
            "error": "red",
            "lk-fg": "#1fd5f9",
            "log.name": Style.null(),
            "log.extra": Style(dim=True),
            "logging.level.notset": Style(dim=True),
            "logging.level.debug": Style(color="cyan"),
            "logging.level.info": Style(color="green"),
            "logging.level.warning": Style(color="yellow"),
            "logging.level.dev": Style(color="blue"),
            "logging.level.error": Style(color="red", bold=True),
            "logging.level.critical": Style(color="red", bold=True, reverse=True),
        }
        self.tag_width = 11
        self.console = Console(theme=Theme(theme))

        self._enabled = False
        self._log_handler = RichLoggingHandler(self)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def print(
        self, child: RenderableType, *, tag: str = "", tag_style: Style | None = None
    ) -> None:
        self.console.print(self._render_tag(child, tag=tag, tag_style=tag_style))

    def _render_tag(
        self,
        child: RenderableType,
        *,
        tag: str = "",
        tag_width: int | None = None,
        tag_style: Style | None = None,
    ) -> ConsoleRenderable:
        if tag:
            tag = f" {tag} "

        tag_width = tag_width or self.tag_width
        table = Table.grid(
            Column(width=tag_width + 2, no_wrap=True),
            Column(no_wrap=False, overflow="fold"),
            padding=(0, 0, 0, 0),
            collapse_padding=True,
            pad_edge=False,
        )

        left_padding = tag_width - len(tag)
        left_padding = max(0, left_padding)

        style = tag_style or self.console.get_style("tag")
        tag_segments = [Segment(tag, style=style)]

        left = [Segment(" " * left_padding), *tag_segments]
        table.add_row(Group(*left), Group(child))  # type: ignore
        return table


class RichLoggingHandler(logging.Handler):
    def __init__(self, agents_console: AgentsConsole):
        super().__init__()
        self.c = agents_console

        # used to avoid rendering two same time
        self._last_time: Text | None = None

    def emit(self, record: logging.LogRecord) -> None:
        def middle_truncate(s: str, max_width: int) -> str:
            if len(s) <= max_width:
                return s
            if max_width <= 1:
                return "…"[:max_width]
            visible = max_width - 1  # leave room for the ellipsis
            left = visible // 2
            right = visible - left
            return s[:left] + "…" + s[-right:]

        has_exc = bool(
            (record.exc_info and record.exc_info != (None, None, None)) or record.exc_text
        )

        if has_exc:
            exc_info, exc_text = record.exc_info, record.exc_text
            record.exc_info = None  # temporarily strip for clean message
            record.exc_text = None
            try:
                message = self.format(record)
            finally:
                record.exc_info, record.exc_text = exc_info, exc_text
        else:
            message = self.format(record)

        MAX_NAME_WIDTH = 18

        output = Table.grid(padding=(0, 1))
        output.add_column(style="log.time")
        output.add_column(style="log.level", width=8, no_wrap=True)
        output.add_column(style="log.name", width=MAX_NAME_WIDTH, no_wrap=True, overflow="ellipsis")
        output.add_column(ratio=1, style="log.message")
        output.add_column(style="log.extra", no_wrap=True)

        row: list[RenderableType] = []

        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.datetime.fromtimestamp(record.created)
        log_time = log_time or self.c.console.get_datetime()

        log_time_display = (
            Text(log_time.strftime(time_format))
            if time_format
            else Text(log_time.strftime("%H:%M:%S.%f")[:-3])
        )

        if log_time_display == self._last_time:
            time_str = log_time_display.plain
            row.append(Text(" " * len(time_str)))
        else:
            row.append(log_time_display)
            self._last_time = log_time_display

        level_text = Text.styled(
            record.levelname.ljust(8),
            f"logging.level.{record.levelname.lower()}",
        )
        row.append(level_text)

        logger_name = middle_truncate(record.name, MAX_NAME_WIDTH)
        name_text = Text(logger_name)
        row.append(name_text)

        msg_text = Text(message)
        row.append(msg_text)

        console_width = self.c.console.width
        tag_width = 2  # matches self.c._render_tag(..., tag_width=2)
        available_width = max(console_width - tag_width - 6, 20)

        time_len = log_time_display.cell_len
        level_len = 8
        name_len = min(name_text.cell_len, 16)
        msg_len = msg_text.cell_len

        extra: dict[Any, Any] = {}
        _merge_record_extra(record, extra)

        extra_str = ""
        extra_len = 0
        if extra:
            extra_str = json.dumps(extra, cls=JsonFormatter.JsonEncoder, ensure_ascii=False)
            extra_text = Text(extra_str)
            extra_len = extra_text.cell_len

        spaces_between_columns = 4
        total_len_with_extra = (
            time_len + level_len + name_len + msg_len + extra_len + spaces_between_columns
        )

        inline_extra = bool(extra_str) and total_len_with_extra <= available_width

        if inline_extra:
            row.append(Text(extra_str, style="log.extra"))
        else:
            row.append(Text(" "))

        output.add_row(*row)
        output = self.c._render_tag(output, tag_width=tag_width)  # type: ignore

        try:
            self.c.console.print(output)

            if extra_str and not inline_extra:
                indent_width = tag_width + time_len + 1 + level_len + 1 + name_len + 1

                indent = " " * (indent_width + 2)
                extra_line = Text(indent + extra_str, style="log.extra")
                self.c.console.print(extra_line)

            if has_exc:
                self._print_plain_traceback(record)

        except Exception:
            self.handleError(record)

    def _print_plain_traceback(self, record: logging.LogRecord) -> None:
        try:
            if record.exc_text:
                tb_str = record.exc_text
            else:
                exc_type, exc_value, exc_tb = record.exc_info  # type: ignore[misc]
                tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

            tb_text = Text(tb_str, style="red")
            self.c.console.print(tb_text, end="")
            self.c.console.print()

        except Exception:
            self.handleError(record)


def _configure_logger(c: AgentsConsole | None, log_level: int | str) -> None:
    logging.addLevelName(TRACE_LOG_LEVEL, "TRACE")

    root = logging.getLogger()
    if c:
        root.addHandler(c._log_handler)
    else:
        handler = logging.StreamHandler(sys.stdout)
        root.addHandler(handler)
        handler.setFormatter(JsonFormatter())

    root.setLevel(log_level)

    _silence_noisy_loggers()

    from ..log import logger

    if logger.level == logging.NOTSET:
        logger.setLevel(log_level)

    from ..plugin import Plugin

    def _configure_plugin_logger(plugin: Plugin) -> None:
        if plugin.logger is not None and plugin.logger.level == logging.NOTSET:
            plugin.logger.setLevel(log_level)

    for plugin in Plugin.registered_plugins:
        _configure_plugin_logger(plugin)

    Plugin.emitter.on("plugin_registered", _configure_plugin_logger)


_CLI_SETUP_DOCS = "https://docs.livekit.io/reference/developer-tools/livekit-cli/#setup"

# The removed commands still accept their old flags so invocations written against
# the previous CLI reach the migration notice instead of a usage error.
_REMOVED_CMD_CONTEXT = {"ignore_unknown_options": True, "allow_extra_args": True}


def _print_removed_notice(command: str) -> None:
    c = AgentsConsole.get_instance()
    c.print(
        f"{command} mode has been removed from the Python CLI. "
        f"Use [bold]lk agent {command}[/bold] instead: {_CLI_SETUP_DOCS}",
        tag="Removed",
        tag_style=Style.parse("black on yellow"),
    )
    raise typer.Exit(code=1)


class LogLevel(str, enum.Enum):
    trace = "TRACE"
    debug = "DEBUG"
    info = "INFO"
    warn = "WARN"
    error = "ERROR"
    critical = "CRITICAL"


def _build_cli(server: AgentServer) -> typer.Typer:
    app = typer.Typer(rich_markup_mode="rich")

    @app.callback(invoke_without_command=True)
    def _default(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            print(ctx.get_help())
            raise typer.Exit()

    _start_log_default = LogLevel(ServerEnvOption.getvalue(server.log_level, False))

    @app.command(context_settings=_REMOVED_CMD_CONTEXT)
    def console() -> None:
        """
        [red]Removed[/red]: use [bold]lk agent console[/bold] instead
        (https://docs.livekit.io/reference/developer-tools/livekit-cli/#setup).
        """
        _print_removed_notice("console")

    @app.command()
    def start(
        *,
        log_level: Annotated[
            LogLevel,
            typer.Option(
                help="Set the log level", case_sensitive=False, envvar="LIVEKIT_LOG_LEVEL"
            ),
        ] = _start_log_default,
        url: Annotated[
            str | None,  # noqa: UP007
            typer.Option(
                help="The WebSocket URL of your LiveKit server or Cloud project.",
                envvar="LIVEKIT_URL",
            ),
        ] = None,
        api_key: Annotated[
            str | None,  # noqa: UP007
            typer.Option(
                help="API key for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_KEY",
            ),
        ] = None,
        api_secret: Annotated[
            str | None,  # noqa: UP007
            typer.Option(
                help="API secret for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_SECRET",
            ),
        ] = None,
        drain_timeout: Annotated[
            int | None,  # noqa: UP007
            typer.Option(
                help="Time in seconds to wait for jobs to finish before shutting down.",
            ),
        ] = None,
        simulation: Annotated[
            bool,
            typer.Option(
                hidden=True,
                help="Run under an agent simulation: the worker load limit is disabled "
                "so runs can saturate the agent. Set by `lk simulate`.",
            ),
        ] = False,
    ) -> None:
        if drain_timeout is not None:
            server.update_options(drain_timeout=drain_timeout)

        _cli._run_worker(
            server=server,
            args=proto.CliArgs(
                log_level=log_level.value,
                url=url,
                api_key=api_key,
                api_secret=api_secret,
                simulation=simulation,
            ),
        )

    @app.command(context_settings=_REMOVED_CMD_CONTEXT)
    def dev() -> None:
        """
        [red]Removed[/red]: use [bold]lk agent dev[/bold] instead
        (https://docs.livekit.io/reference/developer-tools/livekit-cli/#setup).
        """
        _print_removed_notice("dev")

    @app.command()
    def connect(
        *,
        log_level: Annotated[
            LogLevel,
            typer.Option(help="Set the log level", case_sensitive=False),
        ] = LogLevel.debug,
        url: Annotated[
            str | None,  # noqa: UP007
            typer.Option(
                help="The WebSocket URL of your LiveKit server or Cloud project.",
                envvar="LIVEKIT_URL",
            ),
        ] = None,
        api_key: Annotated[
            str | None,  # noqa: UP007
            typer.Option(
                help="API key for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_KEY",
            ),
        ] = None,
        api_secret: Annotated[
            str | None,  # noqa: UP007
            typer.Option(
                help="API secret for authenticating with your LiveKit server or Cloud project.",
                envvar="LIVEKIT_API_SECRET",
            ),
        ] = None,
        room: Annotated[
            str,
            typer.Option(help="Room name to connect to"),
        ],
        participant_identity: Annotated[
            str | None,  # noqa: UP007
            typer.Option(help="Participant identity"),
        ] = None,
    ) -> None:
        if participant_identity is None:
            participant_identity = shortuuid("agent-")

        c = AgentsConsole.get_instance()
        _configure_logger(c, log_level.value)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _task: asyncio.Task | None = None

        @server.once("worker_started")
        def _simulate_job() -> None:
            nonlocal _task

            async def simulate_job() -> None:
                async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
                    room_request = api.ListRoomsRequest(names=[room])
                    active_room = await lk_api.room.list_rooms(room_request)

                    if not active_room.rooms:
                        room_info = await lk_api.room.create_room(api.CreateRoomRequest(name=room))
                    else:
                        room_info = active_room.rooms[0]

                await server.simulate_job(
                    room=room,
                    fake_job=False,
                    room_info=room_info,
                    agent_identity=participant_identity,
                )

            _task = asyncio.create_task(simulate_job())

        try:
            loop.run_until_complete(server.run(devmode=True, unregistered=True))
        except _ExitCli:
            raise typer.Exit() from None
        except KeyboardInterrupt:
            logger.warning("exiting forcefully")
            os._exit(1)
        except (CLIError, ValueError) as e:
            c.print(" ")
            c.print(f"[error]{e}")
            c.print(" ")
            raise typer.Exit(code=1) from None

    @app.command()
    def download_files() -> None:
        import warnings

        c = AgentsConsole.get_instance()
        c.enabled = True

        _configure_logger(c, logging.DEBUG)

        c.print(
            "[yellow]Invoking the download-files command via your agent script is "
            "deprecated as of 1.5.10. Run it directly against the livekit.agents module "
            "instead, e.g. `uv run -m livekit.agents download-files`.[/yellow]"
        )
        warnings.warn(
            "Invoking the download-files command via your agent script is deprecated "
            "as of 1.5.10. Run it directly against the livekit.agents module instead, "
            "e.g. `uv run -m livekit.agents download-files`.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            for plugin in Plugin.registered_plugins:
                logger.info(f"Downloading files for {plugin.package}")
                plugin.download_files()
                logger.info(f"Finished downloading files for {plugin.package}")

        except CLIError as e:
            c.print(" ")
            c.print(f"[error]{e}")
            c.print(" ")
            raise typer.Exit(code=1) from None

    return app


def run_app(server: AgentServer | WorkerOptions) -> None:
    import warnings

    warnings.warn(
        "the built-in Python CLI is deprecated; use the LiveKit CLI (`lk agent ...`) "
        "or `python -m livekit.agents`. It will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(server, WorkerOptions):
        server = AgentServer.from_server_options(server)

    _build_cli(server)()
