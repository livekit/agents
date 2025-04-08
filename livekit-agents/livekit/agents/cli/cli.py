from __future__ import annotations  # noqa: I001


import click

from .. import utils
from ..log import logger
from ..plugin import Plugin
from ..types import NOT_GIVEN, NotGivenOr
from ..worker import JobExecutorType, WorkerOptions, SimulateJobInfo
from . import proto, _run
from .log import setup_logging

CLI_ARGUMENTS: proto.CliArgs | None = None


def run_app(
    opts: WorkerOptions,
    *,
    hot_reload: NotGivenOr[bool] = NOT_GIVEN,
) -> None:
    """Run the CLI to interact with the worker"""

    cli = click.Group()

    @cli.command(help="Start the worker in production mode.")
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Set the logging level",
    )
    @click.option(
        "--url",
        envvar="LIVEKIT_URL",
        help="LiveKit server or Cloud project's websocket URL",
    )
    @click.option(
        "--api-key",
        envvar="LIVEKIT_API_KEY",
        help="LiveKit server or Cloud project's API key",
    )
    @click.option(
        "--api-secret",
        envvar="LIVEKIT_API_SECRET",
        help="LiveKit server or Cloud project's API secret",
    )
    @click.option(
        "--drain-timeout",
        default=60,
        help="Time in seconds to wait for jobs to finish before shutting down",
    )
    def start(log_level: str, url: str, api_key: str, api_secret: str, drain_timeout: int) -> None:
        opts.ws_url = url or opts.ws_url
        opts.api_key = api_key or opts.api_key
        opts.api_secret = api_secret or opts.api_secret
        args = proto.CliArgs(
            opts=opts,
            log_level=log_level,
            devmode=False,
            asyncio_debug=False,
            register=True,
            watch=False,
            drain_timeout=drain_timeout,
        )
        _run.run_worker(args)

    @cli.command(help="Start the worker in development mode")
    @click.option(
        "--log-level",
        default="DEBUG",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Set the logging level",
    )
    @click.option(
        "--url",
        envvar="LIVEKIT_URL",
        help="LiveKit server or Cloud project's websocket URL",
    )
    @click.option(
        "--api-key",
        envvar="LIVEKIT_API_KEY",
        help="LiveKit server or Cloud project's API key",
    )
    @click.option(
        "--api-secret",
        envvar="LIVEKIT_API_SECRET",
        help="LiveKit server or Cloud project's API secret",
    )
    @click.option(
        "--asyncio-debug/--no-asyncio-debug",
        default=False,
        help="Enable debugging feature of asyncio",
    )
    @click.option(
        "--watch/--no-watch",
        default=hot_reload if utils.is_given(hot_reload) else True,
        help="Watch for changes in the current directory and plugins in editable mode",
    )
    def dev(
        log_level: str,
        url: str,
        api_key: str,
        api_secret: str,
        asyncio_debug: bool,
        watch: bool,
    ) -> None:
        opts.ws_url = url or opts.ws_url
        opts.api_key = api_key or opts.api_key
        opts.api_secret = api_secret or opts.api_secret
        args = proto.CliArgs(
            opts=opts,
            log_level=log_level,
            devmode=True,
            asyncio_debug=asyncio_debug,
            watch=watch,
            drain_timeout=0,
            register=True,
        )

        _run.run_dev(args)

    @cli.command(help="Start a new chat")
    @click.option(
        "--url",
        envvar="LIVEKIT_URL",
        help="LiveKit server or Cloud project's websocket URL",
    )
    @click.option(
        "--api-key",
        envvar="LIVEKIT_API_KEY",
        help="LiveKit server or Cloud project's API key",
    )
    @click.option(
        "--api-secret",
        envvar="LIVEKIT_API_SECRET",
        help="LiveKit server or Cloud project's API secret",
    )
    def console(
        url: str,
        api_key: str,
        api_secret: str,
    ) -> None:
        # keep everything inside the same process when using the chat mode
        opts.job_executor_type = JobExecutorType.THREAD
        opts.ws_url = url or opts.ws_url or "ws://localhost:7881/fake_console_url"
        opts.api_key = api_key or opts.api_key or "fake_console_key"
        opts.api_secret = api_secret or opts.api_secret or "fake_console_secret"

        args = proto.CliArgs(
            opts=opts,
            log_level="DEBUG",
            devmode=True,
            asyncio_debug=False,
            watch=False,
            console=True,
            drain_timeout=0,
            register=False,
            simulate_job=SimulateJobInfo(room="mock-console"),
        )
        _run.run_worker(args)

    @cli.command(help="Connect to a specific room")
    @click.option(
        "--log-level",
        default="DEBUG",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Set the logging level",
    )
    @click.option(
        "--url",
        envvar="LIVEKIT_URL",
        help="LiveKit server or Cloud project's websocket URL",
    )
    @click.option(
        "--api-key",
        envvar="LIVEKIT_API_KEY",
        help="LiveKit server or Cloud project's API key",
    )
    @click.option(
        "--api-secret",
        envvar="LIVEKIT_API_SECRET",
        help="LiveKit server or Cloud project's API secret",
    )
    @click.option(
        "--asyncio-debug/--no-asyncio-debug",
        default=False,
        help="Enable debugging feature of asyncio",
    )
    @click.option(
        "--watch/--no-watch",
        default=True,
        help="Watch for changes in the current directory and plugins in editable mode",
    )
    @click.option("--room", help="Room name to connect to", required=True)
    @click.option("--participant-identity", help="Participant identity (JobType.JT_PUBLISHER)")
    def connect(
        log_level: str,
        url: str,
        api_key: str,
        api_secret: str,
        asyncio_debug: bool,
        watch: bool,
        room: str,
        participant_identity: str,
    ) -> None:
        opts.ws_url = url or opts.ws_url
        opts.api_key = api_key or opts.api_key
        opts.api_secret = api_secret or opts.api_secret
        args = proto.CliArgs(
            opts=opts,
            log_level=log_level,
            devmode=True,
            register=False,
            asyncio_debug=asyncio_debug,
            watch=watch,
            drain_timeout=0,
            simulate_job=SimulateJobInfo(room=room, participant_identity=participant_identity),
        )

        _run.run_dev(args)

    @cli.command(help="Download plugin dependency files")
    @click.option(
        "--log-level",
        default="DEBUG",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Set the logging level",
    )
    def download_files(log_level: str) -> None:
        setup_logging(log_level, True, False)

        for plugin in Plugin.registered_plugins:
            logger.info(f"Downloading files for {plugin}")
            plugin.download_files()
            logger.info(f"Finished downloading files for {plugin}")

    cli()
