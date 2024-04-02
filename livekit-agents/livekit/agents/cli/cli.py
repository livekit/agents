import logging
import functools
import asyncio
import logging
import signal
import click

from attrs import define

from .. import aio
from ..log import logger
from ..worker import Worker, WorkerOptions


@define(kw_only=True)
class CliArgs:
    opts: WorkerOptions
    log_level: str
    production: bool
    asyncio_debug: bool


def run_app(opts: WorkerOptions) -> None:
    """Run the CLI to interact with the worker"""

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

    cli = click.Group()

    @cli.command(help="Start the worker")
    @shared_args
    def start(log_level: str, url: str, api_key: str, api_secret: str) -> None:
        opts.ws_url = url or opts.ws_url
        opts.api_key = api_key or opts.api_key
        opts.api_secret = api_secret or opts.api_secret
        args = CliArgs(
            opts=opts, log_level=log_level, production=True, asyncio_debug=False
        )
        _run_worker_blocking(args)

    @cli.command(help="Send a message to the worker")
    @shared_args
    @click.option(
        "--asyncio-debug/--no-asyncio-debug",
        default=False,
        help="Enable debugging feature of asyncio",
    )
    def dev(
        log_level: str, url: str, api_key: str, api_secret: str, asyncio_debug: bool
    ) -> None:
        from watchfiles import run_process

        opts.ws_url = url or opts.ws_url
        opts.api_key = api_key or opts.api_key
        opts.api_secret = api_secret or opts.api_secret
        args = CliArgs(
            opts=opts,
            log_level=log_level,
            production=False,
            asyncio_debug=asyncio_debug,
        )
        _setup_logging(
            args.log_level, args.production
        )  # setup logger for reloader process
        logger.debug("starting development mode", extra={"cli_args": args})
        run_process("./", target=_run_worker_blocking, args=(args,))

    cli()


def _run_worker_blocking(args: CliArgs) -> None:
    class Shutdown(SystemExit):
        pass

    _setup_logging(args.log_level, args.production)

    loop = asyncio.get_event_loop()
    worker = Worker(args.opts, loop=loop)

    loop.set_debug(args.asyncio_debug)
    loop.slow_callback_duration = 0.02  # 20ms
    aio.debug.hook_slow_callbacks(0.75)

    try:

        def _signal_handler():
            raise Shutdown

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    except NotImplementedError:
        # TODO(theomonnom): add_signal_handler is not implemented on win
        pass

    async def _worker_run(
        worker: Worker,
    ) -> None:
        try:
            await worker.run()
        except Exception as e:
            logger.exception("worker failed", exc_info=e)
        finally:
            await worker.aclose()

    main_task = loop.create_task(_worker_run(worker))
    try:
        loop.run_until_complete(main_task)
    except (Shutdown, KeyboardInterrupt):
        pass

    main_task.cancel()
    loop.run_until_complete(main_task)

    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


class ExtraLogFormatter(logging.Formatter):
    def __init__(self, formatter):
        super().__init__()
        self._formatter = formatter

    def format(self, record):
        # less hacky solution?
        dummy = logging.LogRecord("", 0, "", 0, None, None, None)
        extra_txt = "\t "
        for k, v in record.__dict__.items():
            if k not in dummy.__dict__:
                extra_txt += " {}={}".format(k, str(v).replace("\n", " "))
        message = self._formatter.format(record)
        return message + extra_txt


def _setup_logging(log_level: str, production: bool = True) -> None:
    h = logging.StreamHandler()

    if production:
        ## production mode, json logs
        from pythonjsonlogger import jsonlogger

        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super().add_fields(log_record, record, message_dict)
                log_record.pop("taskName")
                log_record["level"] = record.levelname

        formatter = CustomJsonFormatter("%(asctime)s %(level)s %(name)s %(message)s")
        h.setFormatter(formatter)
    else:
        ## dev mode, colored logs & show all extra
        import colorlog

        h.setFormatter(
            ExtraLogFormatter(
                colorlog.ColoredFormatter(
                    "%(asctime)s %(log_color)s%(levelname)-4s %(bold_white)s %(name)s %(reset)s %(message)s",
                    log_colors={
                        **colorlog.default_log_colors,
                        "DEBUG": "blue",
                    },
                )
            )
        )

    logging.root.addHandler(h)
    logging.root.setLevel(log_level)
