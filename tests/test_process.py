import asyncio
import logging
import multiprocessing

from livekit.agents import ipc
from livekit.protocol import agent

TEST_STR = "the people who are crazy enough to think they can change the world are the ones who do"


def _process_target(cch: ipc.protocol.ProcessPipe):
    loop = asyncio.get_event_loop()
    cpipe = ipc.apipe.AsyncPipe(cch, loop)

    async def _run():
        assert await cpipe.read() == ipc.protocol.StartJobRequest(job=agent.Job())
        await cpipe.write(ipc.protocol.StartJobResponse(exc=None))
        await cpipe.write(ipc.protocol.Log(level=logging.INFO, message=TEST_STR))

    loop.run_until_complete(_run())
    cpipe.close()


def test_protocol():
    loop = asyncio.get_event_loop()
    pch, cch = multiprocessing.Pipe(duplex=True)
    proc = multiprocessing.Process(target=_process_target, args=(cch,))
    ppipe = ipc.apipe.AsyncPipe(pch, loop=loop)
    proc.start()

    async def _run():
        await ppipe.write(ipc.protocol.StartJobRequest(job=agent.Job()))
        assert await ppipe.read() == ipc.protocol.StartJobResponse(exc=None)
        assert await ppipe.read() == ipc.protocol.Log(
            level=logging.INFO, message=TEST_STR
        )

    loop.run_until_complete(_run())
    ppipe.close()
    proc.join()
