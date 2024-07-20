import logging

import pytest
from livekit.agents import utils


@pytest.fixture
def job_process(event_loop):
    utils.http_context._new_session_ctx()
    yield
    event_loop.run_until_complete(utils.http_context._close_http_ctx())


@pytest.fixture(autouse=True)
def default_log_level_fixture(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
        yield
