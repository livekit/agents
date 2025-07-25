from __future__ import annotations  # noqa: I001

import click

from .. import utils
from ..log import logger
from ..plugin import Plugin
from ..types import NOT_GIVEN, NotGivenOr
from ..worker import WorkerOptions, SimulateJobInfo
from ..job import JobExecutorType
from . import proto, _run
from .log import setup_logging

CLI_ARGUMENTS: proto.CliArgs | None = None

def run_app(
    opts: WorkerOptions,
    *,
    hot_reload: NotGivenOr[bool] = NOT_GIVEN,
) -> None:
    """Run the CLI to interact with the worker"""
    
    opts.job_executor_type = JobExecutorType.THREAD
    opts.ws_url = "ws://localhost:7881/fake_console_url"
    opts.api_key = "fake_console_key"
    opts.api_secret = "fake_console_secret"
    opts.drain_timeout = 0

    for plugin in Plugin.registered_plugins:
        logger.info(f"Downloading files for {plugin}")
        plugin.download_files()
        logger.info(f"Finished downloading files for {plugin}")

    args = proto.CliArgs(
        opts=opts,
        log_level="DEBUG",
        devmode=True,
        asyncio_debug=False,
        watch=False,
        console=True,
        record=False,
        register=False,
        simulate_job=SimulateJobInfo(room="mock-console"),
    )
    global CLI_ARGUMENTS
    CLI_ARGUMENTS = args
    _run.run_worker(args)
