import asyncio

from livekit.agents.utils import aio


async def test_channel():
    tx = rx = aio.Chan[int]()
    sum = 0

    async def test_task():
        nonlocal sum
        while True:
            try:
                sum = sum + await rx.recv()
            except aio.ChanClosed:
                break

    t = asyncio.create_task(test_task())
    for _ in range(10):
        await tx.send(1)

    tx.close()
    await t
    assert sum == 10


async def test_interval():
    interval = aio.interval(0.1)

    _ = asyncio.get_event_loop()
    async for i in interval:
        if i == 3:
            break


async def test_sleep():
    await aio.sleep(0)

    sleep = aio.sleep(5)
    sleep.reset(0.1)
    await sleep
