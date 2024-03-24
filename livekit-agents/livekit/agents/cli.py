import logging
from typing import List

from .plugin import Plugin
from .worker import Worker, _run_worker


def run_app(worker: Worker) -> None:
    """Run the CLI to interact with the worker"""

    import click

    @click.group()
    @click.option(
        "--log-level",
        default="INFO",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        help="Set the logging level",
    )
    def cli(log_level: str, url: str, api_key: str, api_secret: str) -> None:
        logging.basicConfig(level=log_level)
        worker._set_url(url)
        worker._api_key = api_key
        worker._api_secret = api_secret

    @cli.command(help="Start the worker")
    @click.option(
        "--url",
        required=True,
        envvar="LIVEKIT_URL",
        help="LiveKit server or Cloud project WebSocket URL",
        default="ws://localhost:7880",
    )
    @click.option(
        "--api-key",
        envvar="LIVEKIT_API_KEY",
        help="LiveKit server or Cloud project's API key",
        required=True,
    )
    @click.option(
        "--api-secret",
        envvar="LIVEKIT_API_SECRET",
        help="LiveKit server or Cloud project's API secret",
        required=True,
    )
    def start() -> None:
        _run_worker(worker)

    @cli.command(help="List used plugins")
    def plugins() -> None:
        for plugin in Plugin.registered_plugins:
            logging.info(plugin.title)

    @cli.command(help="Download required files of used plugins")
    @click.option("--exclude", help="Exclude plugins", multiple=True)
    def download_files(exclude: List[str]) -> None:
        for plugin in Plugin.registered_plugins:
            if plugin.title in exclude:
                continue

            logging.info("Setup data for plugin %s", plugin.title)
            plugin.download_files()

    cli()
