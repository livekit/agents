import pytest
from livekit.agents import utils


@pytest.fixture()
async def job_process():
    g_session = utils.http_session()
    yield
    await g_session.close()
