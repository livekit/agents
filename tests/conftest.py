import dataclasses
import logging

import pytest
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, utils
from livekit.agents.cli import log

TEST_CONNECT_OPTIONS = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)


@pytest.fixture
def job_process(event_loop):
    utils.http_context._new_session_ctx()
    yield
    event_loop.run_until_complete(utils.http_context._close_http_ctx())


@pytest.fixture(autouse=True)
def configure_test():
    log._silence_noisy_loggers()


@pytest.fixture()
def logger():
    logger = logging.getLogger("livekit.tests")
    logger.setLevel(logging.DEBUG)
    return logger
