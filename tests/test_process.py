import asyncio
import logging
import multiprocessing
import pickle
from os import environ

from livekit import api, rtc
from livekit.agents.apipe import AsyncPipe
from livekit.agents.ipc.protocol import (
    IPC_MESSAGES,
    Log,
    StartJobRequest,
    StartJobResponse,
)
from livekit.agents.worker import WorkerOptions
from livekit.protocol import agent

TEST_STR = "the people who are crazy enough to think they can change the world are the ones who do"


def _process_protocol_target(cch):
    loop = asyncio.get_event_loop()
    cpipe = AsyncPipe(cch, loop, messages=IPC_MESSAGES)

    async def _run():
        assert await cpipe.read() == StartJobRequest(job=agent.Job())
        await cpipe.write(StartJobResponse(exc=None))
        await cpipe.write(Log(level=logging.INFO, message=TEST_STR))

    loop.run_until_complete(_run())
    cpipe.close()


def test_protocol():
    loop = asyncio.get_event_loop()
    pch, cch = multiprocessing.Pipe(duplex=True)
    proc = multiprocessing.Process(target=_process_protocol_target, args=(cch,))
    ppipe = AsyncPipe(pch, loop=loop, messages=IPC_MESSAGES)
    proc.start()

    async def _run():
        await ppipe.write(StartJobRequest(job=agent.Job()))
        assert await ppipe.read() == StartJobResponse(exc=None)
        assert await ppipe.read() == Log(level=logging.INFO, message=TEST_STR)

    loop.run_until_complete(_run())
    ppipe.close()
    proc.join()


def _process_rtc_target(q: multiprocessing.Queue, url: str, token: str):
    room = rtc.Room()

    async def _run():
        await room.connect(url, token)
        assert room.isconnected()
        await room.local_participant.publish_data(b"hello")
        await asyncio.sleep(1)
        q.put_nowait(room.sid)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run())


def test_rtc():
    jwt = (
        api.AccessToken()
        .with_identity("test_rtc_process")
        .with_grants(api.VideoGrants(room_join=True, room="test_rtc_process"))
        .to_jwt()
    )

    url = environ.get("LIVEKIT_URL")
    if not url:
        raise ValueError("LIVEKIT_URL must be set")

    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_process_rtc_target, args=(q, url, jwt))
    proc.start()

    room_sid: str = q.get(timeout=10)
    assert room_sid.startswith("RM_")

    proc.join()


def _request_fnc(x):
    pass


def test_ser():
    # make sure WorkerOptions is serializable with pickle
    pickle.loads(pickle.dumps(WorkerOptions(request_fnc=_request_fnc)))
