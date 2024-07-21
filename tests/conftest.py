import logging

import pytest
from livekit.agents import utils, log


@pytest.fixture
def job_process(event_loop):
    utils.http_context._new_session_ctx()
    yield
    event_loop.run_until_complete(utils.http_context._close_http_ctx())
