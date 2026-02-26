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


class TestAsyncAtomicCounter:
    async def test_basic_increment_decrement(self):
        counter = aio.AsyncAtomicCounter()
        assert await counter.get() == 0

        assert await counter.increment() == 1
        assert await counter.increment(5) == 6
        assert await counter.decrement() == 5
        assert await counter.decrement(3) == 2

    async def test_initial_value(self):
        counter = aio.AsyncAtomicCounter(initial=10)
        assert await counter.get() == 10

    async def test_set(self):
        counter = aio.AsyncAtomicCounter()
        await counter.set(42)
        assert await counter.get() == 42

    async def test_compare_and_swap_success(self):
        counter = aio.AsyncAtomicCounter(initial=5)
        assert await counter.compare_and_swap(5, 10) is True
        assert await counter.get() == 10

    async def test_compare_and_swap_failure(self):
        counter = aio.AsyncAtomicCounter(initial=5)
        assert await counter.compare_and_swap(99, 20) is False
        assert await counter.get() == 5

    async def test_get_and_reset(self):
        counter = aio.AsyncAtomicCounter(initial=7)
        prev = await counter.get_and_reset()
        assert prev == 7
        assert await counter.get() == 0

    async def test_get_and_reset_custom_value(self):
        counter = aio.AsyncAtomicCounter(initial=15)
        prev = await counter.get_and_reset(reset_value=3)
        assert prev == 15
        assert await counter.get() == 3

    async def test_get_nowait(self):
        counter = aio.AsyncAtomicCounter(initial=99)
        assert counter.get_nowait() == 99
        await counter.increment()
        assert counter.get_nowait() == 100

    async def test_concurrent_increments(self):
        counter = aio.AsyncAtomicCounter()
        num_tasks = 10
        increments_per_task = 100

        async def increment_many():
            for _ in range(increments_per_task):
                await counter.increment()

        await asyncio.gather(*(increment_many() for _ in range(num_tasks)))
        assert await counter.get() == num_tasks * increments_per_task
