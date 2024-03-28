import logging
import click

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
        help="LiveKit server or Cloud project WebSocket URL",
        default="ws://localhost:7880",  # default for a local OSS server
        required=True,
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
        pass

    cli()
