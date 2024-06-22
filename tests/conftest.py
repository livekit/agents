import pytest
from livekit.agents import utils


@pytest.fixture
def job_process(event_loop):
    utils.http_context.new_session_ctx()
    yield
    event_loop.run_until_complete(utils.http_context.close_http_ctx())
