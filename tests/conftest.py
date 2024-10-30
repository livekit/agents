import pytest
from livekit.agents import utils
from livekit.agents.cli import log


@pytest.fixture
def job_process(event_loop):
    utils.http_context._new_session_ctx()
    yield
    event_loop.run_until_complete(utils.http_context._close_http_ctx())


@pytest.fixture(autouse=True)
def configure_test():
    log._silence_noisy_loggers()
