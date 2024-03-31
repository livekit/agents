import asyncio
import logging
import signal

import click

from ..log import logger
from ..worker import Worker


def run_app(worker: Worker) -> None:
    """Run the CLI to interact with the worker"""

    @click.group()
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        help="Set the logging level",
    )
    def cli(log_level: str) -> None:
        logging.basicConfig(level=log_level)

    @cli.command(help="Start the worker")
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
    def start(url: str, api_key: str, api_secret: str) -> None:
        worker._update_opts(ws_url=url, api_key=api_key, api_secret=api_secret)
        _run_worker_blocking(worker)

    cli()


class GracefulShutdown(SystemExit):
    pass


def _signal_handler():
    raise GracefulShutdown


def _run_worker_blocking(worker: Worker) -> None:
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.slow_callback_duration = 0.01
    loop.set_exception_handler(logger.exception)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    except NotImplementedError:
        # add_signal_handler is not implemented on Windows
        pass

    async def _main_task() -> None:
        try:
            await worker.run()
        except Exception:
            logger.exception("error running worker")
        finally:
            await worker.aclose()

    main_task = loop.create_task(_main_task())
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_task)
    except (GracefulShutdown, KeyboardInterrupt):
        pass
        # logging.info("Graceful shutdown worker")
    finally:
        main_task.cancel()
        loop.run_until_complete(main_task)

        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()

        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
