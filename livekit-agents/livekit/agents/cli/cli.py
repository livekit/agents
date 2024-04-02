import logging
import functools
import asyncio
import logging
import signal
import click

from .. import aio
from ..log import logger
from ..worker import Worker


def _setup_logging(log_level: str, production: bool = True) -> None:
    h = logging.StreamHandler()

    if production:
        # in production, log in json format
        from pythonjsonlogger import jsonlogger

        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super().add_fields(log_record, record, message_dict)
                log_record.pop("taskName")
                log_record["level"] = record.levelname

        formatter = CustomJsonFormatter("%(asctime)s %(level)s %(name)s %(message)s")
        h.setFormatter(formatter)
    else:
        # in dev mode, show all extra and add colors
        import colorlog

        formatter = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s%(levelname)-4s %(bold_white)s %(name)s %(reset)s %(message)s",
            log_colors={
                **colorlog.default_log_colors,
                "DEBUG": "blue",
            },
        )

        class ExtraLogFormatter(logging.Formatter):
            def format(self, record):
                # less hacky solution?
                dummy = logging.LogRecord("", 0, "", 0, None, None, None)
                extra_txt = "\t "
                for k, v in record.__dict__.items():
                    if k not in dummy.__dict__:
                        extra_txt += " {}={}".format(k, str(v).replace("\n", " "))
                message = formatter.format(record)
                return message + extra_txt

        h.setFormatter(ExtraLogFormatter())

    logging.root.addHandler(h)
    logging.root.setLevel(log_level)


def run_app(worker: Worker) -> None:
    """Run the CLI to interact with the worker"""
    loop = asyncio.get_event_loop()

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

    def _update_opts(
        worker: Worker, ws_url: str, api_key: str, api_secret: str
    ) -> None:
        worker._update_opts(ws_url=ws_url, api_key=api_key, api_secret=api_secret)

    cli = click.Group()

    @cli.command(help="Start the worker")
    @shared_args
    def start(log_level: str, url: str, api_key: str, api_secret: str) -> None:
        _setup_logging(log_level)
        _update_opts(worker, url, api_key, api_secret)
        _run_worker_blocking(worker)

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
        _setup_logging(log_level, False)
        _update_opts(worker, url, api_key, api_secret)
        loop.set_debug(asyncio_debug)
        _run_worker_blocking(worker)

    cli()


class GracefulShutdown(SystemExit):
    pass


def _signal_handler():
    raise GracefulShutdown


async def _worker_run(worker: Worker) -> None:
    try:
        await worker.run()
    except Exception as e:
        logger.exception("worker failed", exc_info=e)
    finally:
        await worker.aclose()


def _run_worker_blocking(worker: Worker) -> None:
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 0.02  # 20ms
    aio.debug.hook_slow_callbacks(0.75)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    except NotImplementedError:
        # TODO(theomonnom): add_signal_handler is not implemented on win
        pass

    main_task = loop.create_task(_worker_run(worker))
    try:
        loop.run_until_complete(main_task)
    except (GracefulShutdown, KeyboardInterrupt):
        pass

    main_task.cancel()
    loop.run_until_complete(main_task)

    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
