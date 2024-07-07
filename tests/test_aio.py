import asyncio

from livekit.agents.aio import aio


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


async def test_select():
    async def agen():
        i = 0
        while True:
            yield f"agen-{i}"
            raise StopAsyncIteration
            await asyncio.sleep(0.1)
            i += 1
            if i == 5:
                break

    async def coro():
        await asyncio.sleep(0.1)
        return "coro"

    async def task_f():
        await asyncio.sleep(0.3)
        return "task"

    task = asyncio.create_task(task_f())

    future = asyncio.Future()

    async def mark_future():
        await asyncio.sleep(0.2)
        future.set_result("future")

    asyncio.create_task(mark_future())

    sel = aio.select([agen(), coro(), task, future])
    try:
        async for _ in sel:
            pass
    finally:
        await sel.aclose()

    sel = aio.select([agen(), coro()])
    try:
        async for _ in sel:
            pass
    finally:
        await sel.aclose()


async def test_select_timeout():
    sleep1 = asyncio.sleep(0.2)
    sleep2 = asyncio.sleep(0.3)

    selectable = [sleep1, sleep2]
    sel = aio.select(selectable)
    assert (await sel.__anext__()).selected == sleep1
    assert (await sel.__anext__()).selected == sleep2


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
