import asyncio

import pytest
from livekit.agents import utils


@pytest.fixture
def job_process():
    yield


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    utils.http_context._new_session_ctx()
    yield loop
    loop.run_until_complete(utils.http_context._close_http_ctx())
