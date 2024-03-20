import asyncio
from livekit.agents import aio


async def test_channel():
    tx, rx = aio.channel()

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
