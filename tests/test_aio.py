import asyncio

import pytest

from livekit.agents.utils import aio
from livekit.agents.utils.aio.itertools import Tee

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


async def test_channel():
    tx = rx = aio.Chan[int]()
    total = 0

    async def receive() -> None:
        nonlocal total
        while True:
            try:
                total += await rx.recv()
            except aio.ChanClosed:
                break

    t = asyncio.create_task(receive())
    for _ in range(10):
        await tx.send(1)

    tx.close()
    await t
    assert total == 10


@pytest.mark.parametrize("wait_for_senders", [False, True])
async def test_channel_close_with_blocked_senders(wait_for_senders: bool) -> None:
    channel = aio.Chan[int](maxsize=1)
    channel.send_nowait(1)
    started = [asyncio.Event(), asyncio.Event()]

    async def send(value: int, ready: asyncio.Event) -> None:
        ready.set()
        await channel.send(value)

    senders = [asyncio.create_task(send(i + 2, ready)) for i, ready in enumerate(started)]
    try:
        for ready in started:
            await ready.wait()
        channel.close()
        if wait_for_senders:
            for sender in senders:
                with pytest.raises(aio.ChanClosed):
                    await sender

        # both immediate and later shutdown paths may close the same channel
        channel.close()
        for sender in senders:
            with pytest.raises(aio.ChanClosed):
                await sender
        assert channel.recv_nowait() == 1
        with pytest.raises(aio.ChanClosed):
            await channel.recv()
        with pytest.raises(aio.ChanClosed):
            await channel.send(4)
    finally:
        for sender in senders:
            sender.cancel()
        await asyncio.gather(*senders, return_exceptions=True)


async def test_interval() -> None:
    # real time: Interval treats a loop clock of 0.0 (a fresh virtual loop) as "never ticked"
    loop = asyncio.get_running_loop()
    started = loop.time()
    async for i in aio.interval(0.01):
        if i == 3:
            break
    assert loop.time() - started >= 0.03


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_sleep_reset() -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    sleep = aio.sleep(5)
    sleep.reset(0.1)
    await sleep
    assert loop.time() - started == pytest.approx(0.1)


async def test_atomic_counter() -> None:
    counter = aio.AsyncAtomicCounter(initial=5)
    assert counter.get_nowait() == 5
    assert await counter.increment() == 6
    assert await counter.increment(5) == 11
    assert await counter.decrement() == 10
    assert await counter.decrement(3) == 7

    assert await counter.compare_and_swap(99, 20) is False
    assert await counter.compare_and_swap(7, 15) is True
    assert await counter.get_and_reset(reset_value=3) == 15
    assert await counter.get_and_reset() == 3
    await counter.set(42)
    assert await counter.get() == 42


async def test_atomic_counter_concurrent_increments() -> None:
    counter = aio.AsyncAtomicCounter()

    async def increment_many() -> None:
        for _ in range(100):
            await counter.increment()

    await asyncio.gather(*(increment_many() for _ in range(10)))
    assert await counter.get() == 1000


async def _async_iter(items, error=None):
    for item in items:
        yield item
    if error is not None:
        raise error


@pytest.mark.parametrize("items", [[1, 2, 3], []])
async def test_tee(items: list[int]) -> None:
    for peer in Tee(_async_iter(items), n=3):
        assert [item async for item in peer] == items


async def test_tee_exception_propagation() -> None:
    tee = Tee(_async_iter([1, 2], ValueError("upstream error")), n=2)

    for peer in tee:
        result = []
        with pytest.raises(ValueError, match="upstream error"):
            async for item in peer:
                result.append(item)
        assert result == [1, 2]


async def test_tee_exception_concurrent() -> None:
    tee = Tee(_async_iter([1], ValueError("concurrent error")), n=2)
    results: list[list[int]] = [[], []]
    errors: list[BaseException | None] = [None, None]

    async def consume(idx: int, peer) -> None:
        try:
            async for item in peer:
                results[idx].append(item)
        except ValueError as e:
            errors[idx] = e

    await asyncio.gather(consume(0, tee[0]), consume(1, tee[1]))

    assert results == [[1], [1]]
    assert [str(e) for e in errors] == ["concurrent error", "concurrent error"]


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_tee_cancelled_error_does_not_propagate_to_peers() -> None:
    items = [1, 2, 3]
    tee = Tee(_async_iter(items), n=2)
    peer1_results: list[int] = []

    async def consume_peer0() -> None:
        async for _item in tee[0]:
            await asyncio.sleep(1000)  # block after the first item so the task can be cancelled

    async def consume_peer1() -> None:
        async for item in tee[1]:
            peer1_results.append(item)

    task0 = asyncio.create_task(consume_peer0())
    task1 = asyncio.create_task(consume_peer1())
    await asyncio.sleep(0.01)

    task0.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task0

    await asyncio.wait_for(task1, timeout=2.0)
    assert peer1_results == items
