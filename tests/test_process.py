import asyncio
import logging
from livekit.protocol import agent
from livekit.agents.ipc import IPCServer


def _start_server():
    server = IPCServer()
    server.start()
    return server


def fake_usercb():
    pass


async def test_job_process():
    server = _start_server()

    fake_job = agent.Job()
    fake_job.id = "fake_job_id"

    proc = server.new_process(fake_job, "http://localhost", "fake_token", fake_usercb)
    await asyncio.sleep(2)

    await server.aclose()
