import asyncio
import functools
import pathlib
import signal
import sys

import click
from livekit.protocol import models

from .. import aio
from ..log import logger
from ..plugin import Plugin
from ..worker import Worker, WorkerOptions
from . import protocol
from .log import setup_logging


def shared_args(func):
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def shared_dev_args(func):
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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _run_dev(
    opts: WorkerOptions,
    log_level: str,
    url: str,
    api_key: str,
    api_secret: str,
    asyncio_debug: bool,
    watch: bool,
    room: str = "",
    participant_identity: str = "",
):
    opts.ws_url = url or opts.ws_url
    opts.api_key = api_key or opts.api_key
    opts.api_secret = api_secret or opts.api_secret
    args = protocol.CliArgs(
        opts=opts,
        log_level=log_level,
        production=False,
        asyncio_debug=asyncio_debug,
        watch=watch,
        drain_timeout=0,
        room=room,
        participant_identity=participant_identity,
    )

    if watch:
        from .watcher import WatchServer

        setup_logging(log_level, args.production)

        main_file = pathlib.Path(sys.argv[0]).parent
        server = WatchServer(run_worker, main_file, args, watch_plugins=True)
        server.run()
    else:
        run_worker(args)


def run_app(opts: WorkerOptions) -> None:
    """Run the CLI to interact with the worker"""

    cli = click.Group()

    @cli.command(help="Start the worker in production mode.")
    @shared_args
    @click.option(
        "--drain-timeout",
        default=60,
        help="Time in seconds to wait for jobs to finish before shutting down",
    )
    def start(
        log_level: str, url: str, api_key: str, api_secret: str, drain_timeout: int
    ) -> None:
        opts.ws_url = url or opts.ws_url
        opts.api_key = api_key or opts.api_key
        opts.api_secret = api_secret or opts.api_secret
        args = protocol.CliArgs(
            opts=opts,
            log_level=log_level,
            production=True,
            asyncio_debug=False,
            watch=False,
            drain_timeout=drain_timeout,
        )
        run_worker(args)

    @cli.command(help="Start the worker in development mode")
    @shared_args
    @shared_dev_args
    def dev(
        log_level: str,
        url: str,
        api_key: str,
        api_secret: str,
        asyncio_debug: bool,
        watch: bool,
    ) -> None:
        _run_dev(opts, log_level, url, api_key, api_secret, asyncio_debug, watch)

    @cli.command(help="Connect to a specific room")
    @shared_args
    @shared_dev_args
    @click.option("--room", help="Room name to connect to", required=True)
    @click.option(
        "--participant-identity", help="Participant identity (JobType.JT_PUBLISHER)"
    )
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
        _run_dev(
            opts,
            log_level,
            url,
            api_key,
            api_secret,
            asyncio_debug,
            watch,
            room,
            participant_identity,
        )

    @cli.command(help="Download plugin dependency files (i.e. model weights, ...)")
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        help="Set the logging level",
    )
    def download_files(log_level: str) -> None:
        setup_logging(log_level, True)

        for plugin in Plugin.registered_plugins:
            logger.info(f"Downloading files for {plugin}")
            plugin.download_files()
            logger.info(f"Finished downloading files for {plugin}")

    cli()


def run_worker(args: protocol.CliArgs) -> None:
    class Shutdown(SystemExit):
        pass

    setup_logging(args.log_level, args.production)

    loop = asyncio.get_event_loop()
    worker = Worker(args.opts, loop=loop)

    loop.set_debug(args.asyncio_debug)
    loop.slow_callback_duration = 0.05  # 50ms
    aio.debug.hook_slow_callbacks(2)

    if args.room:
        # directly connect to a specific roomj
        @worker.once("worker_registered")
        def _connect_on_register(worker_id: str, server_info: models.ServerInfo):
            logger.info("connecting to room %s", args.room)
            loop.create_task(worker.simulate_job(args.room, args.participant_identity))

    try:

        def _signal_handler():
            raise Shutdown

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    except NotImplementedError:
        # TODO(theomonnom): add_signal_handler is not implemented on win
        pass

    async def _worker_run(worker: Worker) -> None:
        try:
            await worker.run()
        except Exception:
            logger.exception("worker failed")

    watch_client = None
    if args.watch:
        from .watcher import WatchClient

        assert args.cch is not None

        watch_client = WatchClient(worker, args.cch, loop=loop)
        watch_client.start()

    main_task = loop.create_task(_worker_run(worker), name="agent_runner")
    try:
        loop.run_until_complete(main_task)
    except (Shutdown, KeyboardInterrupt):
        pass

    if args.production:
        loop.run_until_complete(worker.drain(timeout=args.drain_timeout))

    loop.run_until_complete(worker.aclose())

    if watch_client:
        loop.run_until_complete(watch_client.aclose())

    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
